//
// Created by Jlisowskyy on 14/11/24.
//

#include "CpuCore.cuh"

#include "../cuda_core/Helpers.cuh"
#include "../cuda_core/RookMap.cuh"

#include "Utils.cuh"
#include "MctsEngine.cuh"
#include "cpu_MoveGen.cuh"
#include "../ported/CpuUtils.h"
#include "ProgressBar.cuh"
#include "GlobalState.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <format>
#include <string_view>
#include <string>
#include <cassert>
#include <complex>

static constexpr uint32_t PROG_BAR_STEP_MS = 50;
static constexpr uint32_t NUM_CPU_WORKERS = 128;

void InitializeRookMap() {
    FancyMagicRookMap hostMap{
        false
    }; /* WORKAROUND: This is a workaround for the fact that the constructor is not constexpr */
    CUDA_ASSERT_SUCCESS(cudaMemcpyToSymbol(G_ROOK_FANCY_MAP_INSTANCE, &hostMap, sizeof(FancyMagicRookMap)));

    cpu::AllocateStacks(5);
}

CpuCore::CpuCore() = default;

CpuCore::~CpuCore() {
    delete m_deviceProps;
    cpu::DeallocStacks(5);
}

void CpuCore::runCVC(const uint32_t moveTime) {
    _runCVC<MctsEngine<EngineType::GPU0>, MctsEngine<EngineType::GPU0> >(moveTime);
}

void CpuCore::runPVC(const uint32_t moveTime, const uint32_t playerColor) {
    /* prepare components */
    cuda_Board board = *m_board;

    std::vector<cuda_Move> moves = ported_translation::GenMoves(board);
    MctsEngine<EngineType::GPU0> engine{board, NUM_CPU_WORKERS};
    uint32_t numMoves{};

    /* Run in loop until moves are exhausted */
    while (!moves.empty()) {
        cuda_Move pickedMove{};

        std::cout << '\n';
        cpu::DisplayBoard(board.DumpToExternal());
        std::cout << "Material eval: " << board.MaterialEval << std::endl;

        /* pick next move */
        if (board.MovingColor == playerColor) {
            pickedMove = _readPlayerMove(moves);

            ClearLines(27);

            std::cout << "Player picked next move: " << pickedMove.GetPackedMove().GetLongAlgebraicNotationCPU()
                    << std::endl;
        } else {
            std::cout << "Engine will be thinking for " << moveTime << " milliseconds!" << std::endl;

            /* Engine move processing */
            engine.MoveSearchStart();
            _runProcessingAnim(moveTime);
            pickedMove = engine.MoveSearchWait();

            if (g_GlobalState.WriteDotFiles) {
                engine.DumpTreeToDOTFile(std::string("tree_out_") + std::to_string(numMoves) + ".dot");
                engine.DumpHeadTreeToDOTFile(std::string("tree_head_out_") + std::to_string(numMoves) + ".dot");
            }

            ClearLines(28);
            engine.DisplayResults(numMoves);
            ++numMoves;
        }

        assert(pickedMove.IsOkayMoveCPU() && "CPUCORE RECEIVED MALFUNCTIONING MOVE!");

        /* apply move */
        cuda_Move::MakeMove(pickedMove, board);
        engine.ApplyMove(pickedMove);

        moves = ported_translation::GenMoves(board);
    }

    /* Decide who won the game */
    if (ported_translation::IsCheck(board)) {
        const uint32_t winningColor = SwapColor(board.MovingColor);
        const std::string winningColorStr = winningColor == WHITE ? "WHITE" : "BLACK";

        std::cout << std::string(80, '-') << std::endl << std::endl;
        std::cout << std::string(25, ' ') << "GAME WON BY " << winningColorStr << std::endl;
        std::cout << std::endl << std::string(80, '-') << std::endl;
    } else {
        std::cout << std::string(80, '-') << std::endl << std::endl;
        std::cout << std::string(35, ' ') << "DRAW" << std::endl;
        std::cout << std::endl << std::string(80, '-') << std::endl;
    }
}

cuda_Move CpuCore::_readPlayerMove(const std::vector<cuda_Move> &correctMoves) {
    static constexpr std::string_view msg = "Please provide move in Long Algebraic Notation (e.g. A1A2):";

    std::string input{};
    cuda_Move outMove{};
    bool isValid = false;
    uint32_t retries{};

    do {
        std::cout << msg << std::endl;

        /* preprocess input */
        std::getline(std::cin, input);
        cpu::Trim(input);
        for (char &c: input) {
            c = std::tolower(c);
        }

        ++retries;

        /* Check if this is a legal move */
        for (const auto &move: correctMoves) {
            if (move.GetPackedMove().GetLongAlgebraicNotationCPU() == input) {
                outMove = move;
                isValid = true;
                break;
            }
        }
    } while (!isValid);

    /* clean our mess from the console */
    ClearLines(retries * 2);
    return outMove;
}

void CpuCore::init() {
    const auto [bestDeviceIdx, deviceHardwareThreads, deviceProps] = _pickGpu();

    assert(deviceProps != nullptr && "Device properties cannot be nullptr");
    CUDA_ASSERT_SUCCESS(cudaSetDevice(bestDeviceIdx));

    size_t stackSize;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::cout << "Current stack size: " << stackSize << " bytes" << std::endl;

    m_deviceThreads = deviceHardwareThreads;
    m_deviceProps = deviceProps;

    InitializeRookMap();
}

void CpuCore::parseFlags(const int argc, const char **argv) {
    for (int idx = 1; idx < argc; ++idx) {
        if (const auto it = g_GlobalStateCommands.find(std::string(argv[idx])); it != g_GlobalStateCommands.end()) {
            it->second();
        }
    }
}

std::tuple<int, int, cudaDeviceProp *> CpuCore::_pickGpu() {
    int deviceCount;

    std::cout << "Processing CUDA devices..." << std::endl;

    CUDA_ASSERT_SUCCESS(cudaGetDeviceCount(&deviceCount));

    std::cout << std::format("Found {} CUDA devices", deviceCount) << std::endl;

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        exit(EXIT_FAILURE);
    }

    int bestDeviceIdx = 0;
    int bestDeviceScore = 0;
    std::string bestDeviceName{};
    cudaDeviceProp *bestDeviceProps = nullptr;
    bestDeviceProps = new cudaDeviceProp;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop{};

        if (CUDA_TRACE_ERROR(cudaGetDeviceProperties(&prop, i))) {
            continue;
        }

        _dumpGPUInfo(i, prop);

        const int MaxThreadsPerSM = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;

        if (MaxThreadsPerSM > bestDeviceScore) {
            bestDeviceScore = MaxThreadsPerSM;
            bestDeviceIdx = i;
            bestDeviceName = prop.name;
        }
    }

    std::cout << std::format("Device chosen for computation: {} ({}) with {} threads available.", bestDeviceName,
                             bestDeviceIdx, bestDeviceScore) << std::endl;

    CUDA_ASSERT_SUCCESS(cudaGetDeviceProperties(bestDeviceProps, bestDeviceIdx));
    return {bestDeviceIdx, bestDeviceScore, bestDeviceProps};
}

void CpuCore::_dumpGPUInfo(const int idx, const cudaDeviceProp &prop) {
    static constexpr std::string_view INFO_FORMAT =
            R"(Device {}: "{}"
          Compute Capability: {}.{}
          Total Global Memory: {:.2f} GB
          Max Threads per Block: {}
          Max Threads Dimensions: [{}, {}, {}]
          Max Grid Size: [{}, {}, {}]
          Max threads per SM: {}
          Number of SMs: {}
          Warp Size: {}
          Memory Clock Rate: {:.0f} MHz
          Memory Bus Width: {} bits
          L2 Cache Size: {} KB
          Max Shared Memory Per Block: {} KB
          Total Constant Memory: {} KB
          Compute Mode: {}
          Concurrent Kernels: {}
          ECC Enabled: {}
          Multi-GPU cpu_Board: {}
          Unified Addressing: {}
)";

    std::cout << std::format(INFO_FORMAT,
                             idx, prop.name,
                             prop.major, prop.minor,
                             static_cast<float>(prop.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f),
                             prop.maxThreadsPerBlock,
                             prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
                             prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2],
                             prop.maxThreadsPerMultiProcessor,
                             prop.multiProcessorCount,
                             prop.warpSize,
                             prop.memoryClockRate * 1e-3f,
                             prop.memoryBusWidth,
                             prop.l2CacheSize / 1024,
                             prop.sharedMemPerBlock / 1024,
                             prop.totalConstMem / 1024,
                             prop.computeMode,
                             prop.concurrentKernels ? "Yes" : "No",
                             prop.ECCEnabled ? "Yes" : "No",
                             prop.isMultiGpuBoard ? "Yes" : "No",
                             prop.unifiedAddressing ? "Yes" : "No"
    ) << std::endl;
}

void CpuCore::setBoard(cuda_Board *board) {
    assert(board != nullptr && "cpu_Board cannot be nullptr");
    m_board = board;
}

bool CpuCore::_validateMove(const std::vector<cuda_Move> &validMoves, const cuda_Move move) {
    return std::ranges::any_of(validMoves,
                               [&](const cuda_Move &m) {
                                   return m.GetPackedMove() == move.GetPackedMove();
                               });
}

void CpuCore::_runProcessingAnim(const uint32_t moveTime) {
    uint32_t timeLeft = moveTime;

    ProgressBar bar(moveTime, 50);
    while (timeLeft) {
        const uint32_t curStep = std::min(PROG_BAR_STEP_MS, timeLeft);

        std::this_thread::sleep_for(std::chrono::milliseconds(curStep));

        bar.Increment(curStep);
        timeLeft -= curStep;
    }
}

void CpuCore::runInfinite() {
    cuda_Board board = *m_board;
    MctsEngine<EngineType::GPU0> engine{board, NUM_CPU_WORKERS};

    auto t1 = std::chrono::steady_clock::now();
    engine.MoveSearchStart();

    _waitForInfiniteStop(engine);

    const auto proposedMove = engine.MoveSearchWait();
    auto t2 = std::chrono::steady_clock::now();

    if (g_GlobalState.WriteDotFiles) {
        engine.DumpTreeToDOTFile("tree_out_infinite.dot");
        engine.DumpHeadTreeToDOTFile("tree_head_out_infinite.dot");
    }

    std::cout << "Engine deduced move: " << proposedMove.GetPackedMove().GetLongAlgebraicNotationCPU() << std::endl;
    std::cout << "After " << (t2 - t1).count() / 1'000'000 << "ms of thinking..." << std::endl;
}

void CpuCore::_waitForInfiniteStop(const MctsEngine<EngineType::GPU0> &engine) {
    static constexpr uint32_t SYNC_INTERVAL = 500;

    std::string input{};

    bool runThread = true;
    ThreadPool moveSyncThread(1);
    moveSyncThread.RunThreads([&](const uint32_t idx) {
        uint32_t timePassedMS{};

        while (runThread) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SYNC_INTERVAL));

            /* Omit double display situation */
            if (!runThread) {
                break;
            }

            timePassedMS += SYNC_INTERVAL;

            std::cout << "[ " << double(timePassedMS) / 1'000.0 << " ] Currently considered move: " <<
                    engine.GetCurrentBestMove().GetPackedMove().GetLongAlgebraicNotationCPU() << std::endl;
        }
    });

    do {
        std::getline(std::cin, input);
        cpu::Trim(input);
        for (char &c: input) {
            c = std::tolower(c);
        }
    } while (input != "stop");

    runThread = false;
    moveSyncThread.Wait();
}

void CpuCore::runCVC1(const uint32_t moveTime) {
    _runCVC<MctsEngine<EngineType::GPU1>, MctsEngine<EngineType::CPU> >(moveTime);
}

void CpuCore::runCVC2(const uint32_t moveTime) {
    _runCVC<MctsEngine<EngineType::GPU0>, MctsEngine<EngineType::GPU1> >(moveTime);
}

template<class ENGINE_T1, class ENGINE_T2>
void CpuCore::_runCVC(const uint32_t moveTime) {
    /* prepare components */
    cuda_Board board = *m_board;

    std::vector<cuda_Move> moves = ported_translation::GenMoves(board);
    ENGINE_T1 engine0{board, ENGINE_T1::GetPreferredThreadsCount()};
    ENGINE_T2 engine1{board, ENGINE_T2::GetPreferredThreadsCount()};
    uint32_t numMoves{};

    /* Pick randomly who should begin */
    uint32_t engineIdx = std::mt19937(std::chrono::steady_clock::now().time_since_epoch().count())() % 2;

    /* Run in loop until moves are exhausted */
    while (!moves.empty()) {
        std::cout << '\n';
        cpu::DisplayBoard(board.DumpToExternal());
        std::cout << "Material eval: " << board.MaterialEval << std::endl;

        std::cout << "Engine will be thinking for " << moveTime << " milliseconds!" << std::endl;

        if (engineIdx == 0) {
            engine0.MoveSearchStart();
        } else {
            engine1.MoveSearchStart();
        }

        _runProcessingAnim(moveTime);
        cuda_Move pickedMove;

        if (engineIdx == 0) {
            pickedMove = engine0.MoveSearchWait();
        } else {
            pickedMove = engine1.MoveSearchWait();
        }

        // ClearLines(29);

        std::cout << "[ " << engineIdx << " | ENGINE: "
                << (engineIdx == 0 ? ENGINE_T1::GetName() : ENGINE_T2::GetName())
                << " ] ";

        if (engineIdx == 0) {
            engine0.DisplayResults(numMoves);

            if (g_GlobalState.WriteDotFiles) {
                engine0.DumpTreeToDOTFile(std::string("tree_out_") + std::to_string(numMoves) + ".dot");
                engine0.DumpHeadTreeToDOTFile(std::string("tree_head_out_") + std::to_string(numMoves) + ".dot");
            }
        } else {
            engine1.DisplayResults(numMoves);

            if (g_GlobalState.WriteDotFiles) {
                engine1.DumpTreeToDOTFile(std::string("tree_out_") + std::to_string(numMoves) + ".dot");
                engine1.DumpHeadTreeToDOTFile(std::string("tree_head_out_") + std::to_string(numMoves) + ".dot");
            }
        }

        assert(pickedMove.IsOkayMoveCPU() && "CPUCORE RECEIVED MALFUNCTIONING MOVE!");

        cuda_Move::MakeMove(pickedMove, board);
        engine0.ApplyMove(pickedMove);
        engine1.ApplyMove(pickedMove);

        /* TODO: fallback when move gen failed? */
        moves = ported_translation::GenMoves(board);

        engineIdx ^= 1;
        ++numMoves;
    }

    /* Decide who won the game */
    if (ported_translation::IsCheck(board)) {
        const uint32_t winningColor = SwapColor(board.MovingColor);
        const std::string winningColorStr = winningColor == WHITE ? "WHITE" : "BLACK";

        std::cout << std::string(80, '-') << std::endl << std::endl;
        std::cout << std::string(25, ' ') << "GAME WON BY " << winningColorStr << std::endl;
        std::cout << std::endl << std::string(80, '-') << std::endl;
    } else {
        std::cout << std::string(80, '-') << std::endl << std::endl;
        std::cout << std::string(35, ' ') << "DRAW" << std::endl;
        std::cout << std::endl << std::string(80, '-') << std::endl;
    }
}
