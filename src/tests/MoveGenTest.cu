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

void TestSinglePositionOutput(const std::string_view &fen) {
    std::cout << "Testing moves of position: " << fen << std::endl;

    const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
    const cuda_Board board(externalBoard);
    auto packedBoard = new SmallPackedBoardT(std::vector{board});

    SmallPackedBoardT *dBoard;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&dBoard, sizeof(SmallPackedBoardT)));

    thrust::device_vector<cuda_Move> dMoves(256);
    thrust::device_vector<int> dCount(1);

    CUDA_ASSERT_SUCCESS(cudaMemcpy(dBoard, packedBoard, sizeof(SmallPackedBoardT), cudaMemcpyHostToDevice));


    GetGPUMovesKernel<<<1, 1>>>(dBoard, thrust::raw_pointer_cast(dMoves.data()),
                                thrust::raw_pointer_cast(dCount.data()));
    CUDA_TRACE_ERROR(cudaGetLastError());
    GUARDED_SYNC();

    const auto cMoves = cpu::GenerateMoves(externalBoard);

    thrust::host_vector<cuda_Move> hMoves = dMoves;
    thrust::host_vector<int> hCount = dCount;

    if (cMoves.size() != hCount[0]) {
        std::cerr << "Moves count mismatch: " << cMoves.size() << " != " << hCount[0] << std::endl;
    }

    const size_t size = std::min(cMoves.size(), static_cast<size_t>(hCount[0]));

    __uint64_t errors = cMoves.size() - size;
    for (size_t i = 0; i < size; ++i) {
        const auto &cMove = cMoves[i];
        cuda_PackedMove ccMove{cMove[0]};
        const auto &hMove = hMoves[i];

        if (hMove.GetPackedMove() != ccMove) {
            std::cerr << "Packed move mismatch device:" << hMove.GetPackedMove().GetLongAlgebraicNotation() << " != "
                      << " host: " << ccMove.GetLongAlgebraicNotation() << std::endl;

            errors++;
        }

        /* Prohibit unused bits */
        static constexpr __uint32_t CheckTypeBit = (__uint32_t) 1 << 15;
        static constexpr __uint32_t PackedIndexesMask = ~(CheckTypeBit);

        if (hMove.GetPackedIndexes() != (PackedIndexesMask & cMove[1])) {
            std::cerr << "Indexes mismatch device:" << u16d(hMove.GetPackedIndexes()) << " != "
                      << " host: " << u16d(cMove[1]) << std::endl;

            std::cerr << "Device: " << hMove.GetPackedMove().GetLongAlgebraicNotation() << std::endl;
            std::cerr << "Host: " << ccMove.GetLongAlgebraicNotation() << std::endl;

            errors++;
        }

        if (hMove.GetPackedMisc() != cMove[2]) {
            std::cerr << "Misc mismatch device:" << u16d(hMove.GetPackedMisc()) << " != "
                      << " host: " << u16d(cMove[2]) << std::endl;

            std::cerr << "Device: " << hMove.GetPackedMove().GetLongAlgebraicNotation() << std::endl;
            std::cerr << "Host: " << ccMove.GetLongAlgebraicNotation() << std::endl;

            errors++;
        }
    }

    if (errors != 0) {
        std::cerr << "Failed test for position: " << fen << std::endl;
        std::cerr << "Errors: " << errors << std::endl;

        std::cerr << "Device moves: " << std::endl;
        for (size_t i = 0; i < size; ++i) {
            std::cerr << hMoves[i].GetPackedMove().GetLongAlgebraicNotation() << std::endl;
        }

        std::cerr << "Host moves: " << std::endl;
        for (size_t i = 0; i < size; ++i) {
            std::cerr << cuda_PackedMove(cMoves[i][0]).GetLongAlgebraicNotation() << std::endl;
        }
    } else {
        std::cout << "Test passed for position: " << fen << std::endl;
    }

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

void FenDBTest(bool fullBruteForceTest = false) {
    auto fens = LoadFenDb();

    for (const auto &fen: fens) {
        TestSinglePositionOutput(fen);
    }

    if (!fullBruteForceTest) {
        return;
    }

    for (const auto &fen: fens) {
        TestMoveCount(fen, TEST_DEPTH);
    }
}

void MoveGenTest_([[maybe_unused]] __uint32_t threadsAvailable, [[maybe_unused]] const cudaDeviceProp &deviceProps) {
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    std::cout << "MoveGen Test" << std::endl;

    std::cout << "Testing positions with depth " << TEST_DEPTH << std::endl;
    for (const auto &fen: TestFEN) {
        TestSinglePositionOutput(fen);
    }

    std::cout << "Testing positions with non zero depth" << std::endl;
    for (auto fen: TestFEN) {
        TestMoveCount(fen, TEST_DEPTH);
    }
    cudaDeviceSetLimit(cudaLimitStackSize, 1024);

    std::cout << "Testing split move gen:" << std::endl;
    MoveGenSplit();

    std::cout << "Testing move gen with whole fen db..." << std::endl;
    FenDBTest();
}

void MoveGenTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    std::cout << sizeof(DefaultPackedBoardT) << std::endl;

    try {
        MoveGenTest_(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Failed test with Error: " << e.what() << std::endl;
    }
}
