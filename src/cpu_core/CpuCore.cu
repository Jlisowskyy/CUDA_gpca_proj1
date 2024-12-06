//
// Created by Jlisowskyy on 14/11/24.
//

#include "CpuCore.cuh"

#include "../cuda_core/Helpers.cuh"
#include "../cuda_core/RookMap.cuh"

#include "Utils.cuh"
#include "MctsEngine.cuh"
#include "cpu_MoveGen.cuh"
#include "CpuUtils.h"
#include "ProgressBar.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <format>
#include <string_view>
#include <string>
#include <cassert>

static constexpr __uint32_t PROG_BAR_STEP_MS = 50;
static constexpr __uint32_t NUM_CPU_WORKERS = 256;

void InitializeRookMap() {
    FancyMagicRookMap hostMap{
            false}; /* WORKAROUND: This is a workaround for the fact that the constructor is not constexpr */
    CUDA_ASSERT_SUCCESS(cudaMemcpyToSymbol(G_ROOK_FANCY_MAP_INSTANCE, &hostMap, sizeof(FancyMagicRookMap)));
}

CpuCore::CpuCore() = default;

CpuCore::~CpuCore() {
    delete m_deviceProps;
}

void CpuCore::runCVC(const __uint32_t moveTime) {
    /* prepare components */
    cuda_Board board = *m_board;

    std::vector<cuda_Move> moves = ported_translation::GenMoves(board);
    MctsEngine<DEFAULT_MCTS_BATCH_SIZE> engine0{board, NUM_CPU_WORKERS};
    MctsEngine<DEFAULT_MCTS_BATCH_SIZE> engine1{board, NUM_CPU_WORKERS};

    MctsEngine<DEFAULT_MCTS_BATCH_SIZE> *engines[]{&engine0, &engine1};
    __uint32_t engineIdx = 0;

    /* Run in loop until moves are exhausted */
    while (!moves.empty()) {
        std::cout << '\n';
        cpu::DisplayBoard(board.DumpToExternal());

        auto &engine = *engines[engineIdx];

        std::cout << "Engine will be thinking for " << moveTime << " milliseconds!" << std::endl;

        engine.MoveSearchStart();
        _runProcessingAnim(moveTime);
        const auto pickedMove = engine.MoveSearchWait();

        ClearLines(28);

        std::cout << "[ ENGINE " << engineIdx << " ] ";
        engine.DisplayResults();

        assert(pickedMove.IsOkayMoveCPU() && "CPUCORE RECEIVED MALFUNCTIONING MOVE!");

        cuda_Move::MakeMove(pickedMove, board);
        engine0.ApplyMove(pickedMove);
        engine1.ApplyMove(pickedMove);

        /* TODO: fallback when move gen failed? */
        moves = ported_translation::GenMoves(board);

        engineIdx ^= 1;
    }

    /* Decide who won the game */
    if (ported_translation::IsCheck(board)) {
        const __uint32_t winningColor = SwapColor(board.MovingColor);
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

void CpuCore::runPVC(const __uint32_t moveTime, const __uint32_t playerColor) {
    /* prepare components */
    cuda_Board board = *m_board;

    std::vector<cuda_Move> moves = ported_translation::GenMoves(board);
    MctsEngine<DEFAULT_MCTS_BATCH_SIZE> engine{board, NUM_CPU_WORKERS};

    /* Run in loop until moves are exhausted */
    while (!moves.empty()) {
        cuda_Move pickedMove{};

        std::cout << '\n';
        cpu::DisplayBoard(board.DumpToExternal());

        /* pick next move */
        if (board.MovingColor == playerColor) {
            pickedMove = _readPlayerMove(moves);

            ClearLines(26);

            std::cout << "Player picked next move: " << pickedMove.GetPackedMove().GetLongAlgebraicNotation()
                      << std::endl;
        } else {
            std::cout << "Engine will be thinking for " << moveTime << " milliseconds!" << std::endl;

            /* Engine move processing */
            engine.MoveSearchStart();
            _runProcessingAnim(moveTime);
            pickedMove = engine.MoveSearchWait();

            ClearLines(28);
            engine.DisplayResults();
        }

        assert(pickedMove.IsOkayMoveCPU() && "CPUCORE RECEIVED MALFUNCTIONING MOVE!");

        /* apply move */
        cuda_Move::MakeMove(pickedMove, board);
        engine.ApplyMove(pickedMove);

        moves = ported_translation::GenMoves(board);
    }

    /* Decide who won the game */
    if (ported_translation::IsCheck(board)) {
        const __uint32_t winningColor = SwapColor(board.MovingColor);
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
    __uint32_t retries{};

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
            if (move.GetPackedMove().GetLongAlgebraicNotation() == input) {
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
    return std::any_of(validMoves.begin(), validMoves.end(),
                       [&](const cuda_Move &m) {
                           return m.GetPackedMove() == move.GetPackedMove();
                       });
}

void CpuCore::_runProcessingAnim(__uint32_t moveTime) {
    __uint32_t timeLeft = moveTime;

    ProgressBar bar(moveTime, 50);
    while (timeLeft) {
        const __uint32_t curStep = std::min(PROG_BAR_STEP_MS, timeLeft);

        std::this_thread::sleep_for(std::chrono::milliseconds(curStep));

        bar.Increment(curStep);
        timeLeft -= curStep;
    }
}

void CpuCore::runInfinite() {
    cuda_Board board = *m_board;
    MctsEngine<DEFAULT_MCTS_BATCH_SIZE> engine{board, NUM_CPU_WORKERS};

    auto t1 = std::chrono::steady_clock::now();
    engine.MoveSearchStart();

    _waitForInfiniteStop(engine);

    const auto proposedMove = engine.MoveSearchWait();
    auto t2 = std::chrono::steady_clock::now();

    std::cout << "Engine deduced move: " << proposedMove.GetPackedMove().GetLongAlgebraicNotation() << std::endl;
    std::cout << "After " << (t2 - t1).count() / 1'000'000 << "ms of thinking..." << std::endl;
}

void CpuCore::_waitForInfiniteStop(MctsEngine<DEFAULT_MCTS_BATCH_SIZE> &engine) {
    static constexpr __uint32_t SYNC_INTERVAL = 500;

    std::string input{};

    bool runThread = true;
    ThreadPool moveSyncThread(1);
    moveSyncThread.RunThreads([&](const __uint32_t idx) {
        __uint32_t timePassedMS{};

        while (runThread) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SYNC_INTERVAL));

            /* Omit double display situation */
            if (!runThread) {
                break;
            }

            timePassedMS += SYNC_INTERVAL;

            std::cout << "[ " << double(timePassedMS) / 1'000.0 << " ] Currently considered move: " <<
                      engine.GetCurrentBestMove().GetPackedMove().GetLongAlgebraicNotation() << std::endl;
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
