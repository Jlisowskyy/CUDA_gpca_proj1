//
// Created by Jlisowskyy on 18/11/24.
//

#include "CudaTests.cuh"

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"
#include "../cuda_core/MoveGenerator.cuh"
#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/ComputeKernels.cuh"
#include "../cuda_core/cuda_PackedBoard.cuh"
#include "../cpu_core/ProgressBar.cuh"

#include "../ported/CpuTests.h"
#include "../ported/CpuUtils.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <string>
#include <string_view>
#include <bitset>
#include <array>
#include <cassert>
#include <filesystem>
#include <random>
#include <chrono>
#include <format>
#include <vector>
#include <map>

using u16d = std::bitset<16>;

static constexpr std::array TestFEN{
        "3r2k1/B7/4q3/1R4P1/1P3r2/8/2P2P1p/R4K2 w - - 0 49",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "3rr1k1/1pq2pp1/p2nb2p/2pp4/6PR/2PBPN2/PPQ2PP1/K2R4 b - - 0 20",
        "7k/r2q1ppp/1p1p4/p1bPrPPb/P1PNPR1P/1PQ5/2B5/R5K1 w - - 23 16",
};

static constexpr __uint32_t TEST_DEPTH = 3;

#define DUMP_MSG                          \
    if (progBar != nullptr) {             \
        progBar->WriteLine(msg);          \
    } else {                              \
        std::cout << msg << std::endl;    \
    }                                     \

#ifdef WRITE_OUT

static constexpr bool WRITE_OUT = true;

#else

static constexpr bool WRITE_OUT = false;

#endif


void __global__ GetGPUMovesKernel(SmallPackedBoardT *board, cuda_Move *outMoves, int *outCount) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ stub sharedStack[DEFAULT_STACK_SIZE];
    __shared__ __uint32_t counter;
    Stack<cuda_Move> stack((cuda_Move*)sharedStack, &counter);

    MoveGenerator<1> gen{(*board)[0], stack};

    gen.GetMovesFast();
    *outCount = static_cast<int>(stack.Size());

    for (int i = 0; i < stack.Size(); ++i) {
        outMoves[i] = stack[i];
    }
}

void __global__ GetGPUMovesKernelSplit(SmallPackedBoardT *board, cuda_Move *outMoves, int *outCount) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    const __uint64_t figIdx = threadIdx.x % 6;
    __shared__ stub sharedStack[DEFAULT_STACK_SIZE];
    __shared__ __uint32_t counter;
    Stack<cuda_Move> stack((cuda_Move *) sharedStack, &counter, false);

    if (figIdx == 0) {
        stack.Clear();
    }

    __syncthreads();
    MoveGenerator<1> gen{(*board)[0], stack};

    gen.GetMovesSplit(figIdx);

    __syncthreads();

    if (figIdx == 0) {
        *outCount = static_cast<int>(stack.Size());

        for (int i = 0; i < stack.Size(); ++i) {
            outMoves[i] = stack[i];
        }
    }
}

void RunBaseKernel(SmallPackedBoardT *dBoard, cuda_Move *dMoves, int *dCount) {
    GetGPUMovesKernel<<<1, 1>>>(dBoard, dMoves, dCount);
};

void RunSplitKernel(SmallPackedBoardT *dBoard, cuda_Move *dMoves, int *dCount) {
    GetGPUMovesKernelSplit<<<1, 6>>>(dBoard, dMoves, dCount);
};


void __global__ GetGPUMoveCountsKernel(SmallPackedBoardT *board, const int depth, __uint64_t *outCount) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ stub sharedStack[10][DEFAULT_STACK_SIZE];
    Stack<cuda_Move> stack(sharedStack);
    MoveGenerator<1> gen{(*board)[0], stack};

    const auto moves = gen.CountMoves(depth, sharedStack);
    *outCount = moves;
}

void TestMoveCount(const std::string_view &fen, int depth) {
    std::cout << "Testing position: " << fen << std::endl;

    const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
    cuda_Board board(externalBoard);

    thrust::device_vector<__uint64_t> dCount(1);
    thrust::device_vector<cuda_Move> dStack(16384);

    auto *packedBoard = new SmallPackedBoardT(std::vector{board});
    SmallPackedBoardT *dBoards;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&dBoards, sizeof(SmallPackedBoardT)));

    GetGPUMoveCountsKernel<<<1, 1>>>(dBoards, depth, thrust::raw_pointer_cast(dCount.data()));
    CUDA_TRACE_ERROR(cudaGetLastError());
    GUARDED_SYNC();

    thrust::host_vector<__uint64_t> hdCount = dCount;
    const __uint64_t hCount = hdCount[0];

    const auto cCount = cpu::CountMoves(externalBoard, depth);

    if (cCount != hCount) {
        std::cerr << "Moves count mismatch: " << cCount << " != " << hCount << std::endl;
        std::cerr << "Depth: " << depth << std::endl;
        std::cerr << "Position: " << fen << std::endl;
    } else {
        std::cout << "Test passed for position: " << fen << std::endl;
    }

    CUDA_ASSERT_SUCCESS(cudaFree(dBoards));
    delete packedBoard;
}

void ValidateMoves(const std::string &fen,
                   const std::vector<cpu::external_move> &cMoves,
                   const std::vector<cuda_Move> &hMoves,
                   ProgressBar *progBar = nullptr,
                   bool writeToOut = false) {

    if (cMoves.size() != hMoves.size()) {
        const auto msg = std::format("GPU move gen failed: Moves count mismatch: {} != {}", cMoves.size(), hCount[0]);
        DUMP_MSG
    }

    std::map<std::string, cuda_Move> GPUMovesMap{};
    for (const auto move: hMoves) {
        const auto [_, result] = GPUMovesMap.emplace(move.GetPackedMove().GetLongAlgebraicNotation(), move);

        assert(result && "GPU move gen failed: received repeated move");
    }

    __uint32_t errors{};
    for (const auto CPUmove: cMoves) {
        cuda_PackedMove convertedCPUMove{CPUmove[0]};

        const auto it = GPUMovesMap.find(convertedCPUMove.GetLongAlgebraicNotation());

        if (it == GPUMovesMap.end()) {
            const auto msg = std::format("GPU move gen failed: {} was not generated by GPU",
                                         convertedCPUMove.GetLongAlgebraicNotation());
            DUMP_MSG

            ++errors;
            continue;
        }

        const cuda_Move gpuMove = it->second;

        if (gpuMove.GetPackedMove() != convertedCPUMove) {
            const auto msg = std::format("GPU move gen failed: Packed move mismatch device: {} != host {}",
                                         gpuMove.GetPackedMove().GetLongAlgebraicNotation(),
                                         convertedCPUMove.GetLongAlgebraicNotation());
            DUMP_MSG

            errors++;
        }

        /* Prohibit unused bits */
        static constexpr __uint32_t CheckTypeBit = (__uint32_t) 1 << 15;
        static constexpr __uint32_t PackedIndexesMask = ~(CheckTypeBit);

        if (gpuMove.GetPackedIndexes() != (PackedIndexesMask & CPUmove[1])) {
            const auto msg = std::format(
                    "GPU move gen failed: Indexes mismatch device: {} != host {}\nDevice: {}\nHost: {}",
                    u16d(gpuMove.GetPackedIndexes()),
                    u16d(CPUmove[1]),
                    gpuMove.GetPackedMove().GetLongAlgebraicNotation(),
                    convertedCPUMove.GetLongAlgebraicNotation());

            DUMP_MSG

            errors++;
        }

        if (gpuMove.GetPackedMisc() != CPUmove[2]) {
            const auto msg = std::format(
                    "GPU move gen failed: Misc mismatch device: {} != host: {}\nDevice: {}\nHost: {}",
                    u16d(gpuMove.GetPackedMisc()),
                    u16d(CPUmove[2]),
                    gpuMove.GetPackedMove().GetLongAlgebraicNotation(),
                    convertedCPUMove.GetLongAlgebraicNotation()
            );

            DUMP_MSG

            errors++;
        }

        GPUMovesMap.erase(convertedCPUMove.GetLongAlgebraicNotation());
    }

    for (const auto &[move, _]: GPUMovesMap) {
        const auto msg = std::format("GPU move gen failed: generated additional move: {}", move);

        DUMP_MSG
    }


    if (errors != 0) {
        std::string msg{};
        msg += "Failed test for position: " + std::string(fen) + '\n';
        msg += "Errors: " + std::to_string(errors) + '\n';
        msg += "Device moves:\n";

        for (const auto &move: hMoves) {
            msg += move.GetPackedMove().GetLongAlgebraicNotation() + '\n';
        }

        msg += "Host moves:\n";
        for (const auto &move: cMoves) {
            msg += cuda_PackedMove(move[0]).GetLongAlgebraicNotation() + '\n';
        }

        DUMP_MSG
    } else {
        const auto msg = std::format("Test passed for position: {}", fen);
        DUMP_MSG
    }
}


template<class FuncT>
void TestSinglePositionOutput(FuncT func, const std::string &fen, ProgressBar *progBar = nullptr,
                              bool writeToOut = false) {

    /* welcome message */
    if (writeToOut) {
        const auto msg = std::format("Testing moves of position: {}", fen);
        DUMP_MSG
    }

    /* Load boards */
    const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
    const cuda_Board board(externalBoard);
    auto packedBoard = new SmallPackedBoardT(std::vector{board});

    SmallPackedBoardT *dBoard;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&dBoard, sizeof(SmallPackedBoardT)));

    /* Load resources to GPU */
    thrust::device_vector<cuda_Move> dMoves(256);
    thrust::device_vector<int> dCount(1);

    CUDA_ASSERT_SUCCESS(cudaMemcpy(dBoard, packedBoard, sizeof(SmallPackedBoardT), cudaMemcpyHostToDevice));

    /* Generate Moves */
    func(dBoard, thrust::raw_pointer_cast(dMoves.data()), thrust::raw_pointer_cast(dCount.data()));

    CUDA_TRACE_ERROR(cudaGetLastError());
    GUARDED_SYNC();

    const auto cMoves = cpu::GenerateMoves(externalBoard);

    thrust::host_vector<cuda_Move> hMoves = dMoves;
    thrust::host_vector<int> hCount = dCount;

    std::vector<cuda_Move> cudaMoves{};
    cudaMoves.reserve(hCount[0]);
    for (int i = 0; i < hCount[0]; ++i) {
        cudaMoves.push_back(hMoves[i]);
    };

    /* validate returned moves */
    ValidateMoves(std::string(fen), cMoves, cudaMoves, progBar, writeToOut);

    CUDA_ASSERT_SUCCESS(cudaFree(dBoard));
    delete packedBoard;
}

void MoveGenSplit() {
//    static constexpr __uint32_t MAX_DEPTH = 32;
//
//    const auto fenDb = LoadFenDb();
//    const auto seeds = GenSeeds(WARP_SIZE);
//
//    std::vector<cuda_Board> boards(WARP_SIZE);
//
//    for (__uint32_t i = 0; i < WARP_SIZE; ++i) {
//        boards[i] = cuda_Board(cpu::TranslateFromFen(fenDb[i]));
//    }
//
//    thrust::device_vector<__uint32_t> dSeeds = seeds;
//    thrust::device_vector<__uint64_t> dResults1(WARP_SIZE);
//    thrust::device_vector<cuda_Move> dMoves(WARP_SIZE * 256);
//
//    thrust::device_vector<DefaultPackedBoardT> dBoards{DefaultPackedBoardT(boards)};
//    SimulateGamesKernelSplitMoves<<<1, BIT_BOARDS_PER_COLOR * WARP_SIZE>>>(thrust::raw_pointer_cast(dBoards.data()),
//                                                                           thrust::raw_pointer_cast(dSeeds.data()),
//                                                                           thrust::raw_pointer_cast(dResults1.data()),
//                                                                           thrust::raw_pointer_cast((dMoves.data())),
//                                                                           MAX_DEPTH);
//    CUDA_TRACE_ERROR(cudaGetLastError());
//    GUARDED_SYNC();
//    thrust::host_vector<__uint64_t> hResults1 = dResults1;
//
//    thrust::device_vector<__uint64_t> dResults2(WARP_SIZE);
//    dBoards = thrust::device_vector<DefaultPackedBoardT>{DefaultPackedBoardT(boards)};
//    SimulateGamesKernel<<<1, WARP_SIZE>>>(thrust::raw_pointer_cast(dBoards.data()),
//                                          thrust::raw_pointer_cast(dSeeds.data()),
//                                          thrust::raw_pointer_cast(dResults2.data()),
//                                          thrust::raw_pointer_cast((dMoves.data())), MAX_DEPTH);
//    CUDA_TRACE_ERROR(cudaGetLastError());
//    GUARDED_SYNC();
//    thrust::host_vector<__uint64_t> hResults2 = dResults2;
//
//    for (__uint32_t i = 0; i < WARP_SIZE; ++i) {
//        if (hResults1[i] != hResults2[i]) {
//            std::cerr << "Split move gen mismatch: " << hResults1[i] << " vs " << hResults2[i] << std::endl;
//            throw std::runtime_error("SPLIT MOVE GEN FAILED THE RESULTS!");
//        }
//    }
}


template<class FuncT>
void RunSinglePosTest(FuncT func) {
    std::cout << "Starting position comparison test for move gen..." << std::endl;

    auto fens = LoadFenDb();

    ProgressBar progBar(TestFEN.size() + fens.size(), 100);
    progBar.Start();

    progBar.WriteLine("Testing move gen with specialized positions");
    for (const auto &fen: TestFEN) {
        TestSinglePositionOutput(func, fen, &progBar, WRITE_OUT);
        progBar.Increment();
    }

    progBar.WriteLine("Testing move gen with whole fen db...");
    for (const auto &fen: fens) {
        TestSinglePositionOutput(func, fen, &progBar, WRITE_OUT);
        progBar.Increment();
    }
}

template<class FuncT>
void RunDepthPosTest(FuncT func) {
    std::cout << "Testing positions with non zero depth" << std::endl;
    for (auto fen: TestFEN) {
        TestMoveCount(fen, TEST_DEPTH);
    }
}

void MoveGenTest_([[maybe_unused]] __uint32_t threadsAvailable, [[maybe_unused]] const cudaDeviceProp &deviceProps) {
    static constexpr __uint32_t EXTENDED_THREAD_STACK_SIZE = 16384;
    static constexpr __uint32_t DEFAULT_THREAD_STACK_SIZE = 1024;

    std::cout << "MoveGen Test" << std::endl;

    std::cout << "Extending thread stack size to " << EXTENDED_THREAD_STACK_SIZE << "..." << std::endl;
    cudaDeviceSetLimit(cudaLimitStackSize, EXTENDED_THREAD_STACK_SIZE);

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Testing plain move gen: " << std::endl;

    RunSinglePosTest(RunBaseKernel);
    std::cout << std::string(80, '-') << std::endl;
//    RunDepthPosTest()

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Testing split move gen: " << std::endl;

//    RunSinglePosTest(RunSplitKernel);
    std::cout << std::string(80, '-') << std::endl;
//    RunDepthPosTest()

    std::cout << std::string(80, '=') << std::endl;


    std::cout << "Reverting thread stack size to " << DEFAULT_THREAD_STACK_SIZE << "..." << std::endl;
    cudaDeviceSetLimit(cudaLimitStackSize, DEFAULT_THREAD_STACK_SIZE);
}

void MoveGenTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    std::cout << sizeof(DefaultPackedBoardT) << std::endl;

    try {
        MoveGenTest_(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Failed test with Error: " << e.what() << std::endl;
    }
}
