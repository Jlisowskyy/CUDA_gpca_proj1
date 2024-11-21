//
// Created by Jlisowskyy on 18/11/24.
//

#include "CudaTests.cuh"

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"
#include "../cuda_core/MoveGenerator.cuh"
#include "../cuda_core/cuda_Board.cuh"

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

static constexpr std::array TestDepths{
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};

static_assert(TestDepths.size() == TestFEN.size());

void __global__ GetGPUMovesKernel(const cuda_Board *board, cuda_Move *outMoves, int *outCount) {
    MoveGenerator gen{*board};

    const auto moves = gen.GetMovesFast();
    *outCount = static_cast<int>(moves.size);

    for (int i = 0; i < moves.size; ++i) {
        outMoves[i] = moves[i];
    }
}

void __global__ GetGPUMoveCounts(const cuda_Board *board, int depth, __uint64_t *outCount) {
    MoveGenerator gen{*board};

    const auto moves = gen.CountMoves(depth);
    *outCount = moves;
}

void TestMoveCount(const std::string_view &fen, int depth) {
    std::cout << "Testing position: " << fen << std::endl;

    const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
    const cuda_Board board(externalBoard);

    thrust::device_vector<cuda_Board> dBoard{board};
    thrust::device_vector<__uint64_t> dCount(1);

    CUDA_ASSERT_SUCCESS(cudaDeviceSetLimit(cudaLimitStackSize, 16384));

    GetGPUMoveCounts<<<1, 1>>>(thrust::raw_pointer_cast(dBoard.data()), depth, thrust::raw_pointer_cast(dCount.data()));

    const auto rc = cudaDeviceSynchronize();
    CUDA_TRACE_ERROR(rc);

    CUDA_ASSERT_SUCCESS(cudaDeviceSetLimit(cudaLimitStackSize, 1024));

    if (rc != cudaSuccess) {
        throw std::runtime_error("Failed to launch kernel");
    }

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
}

void TestSinglePositionOutput(const std::string_view &fen) {
    std::cout << "Testing moves of position: " << fen << std::endl;

    const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
    const cuda_Board board(externalBoard);

    thrust::device_vector<cuda_Move> dMoves(256);
    thrust::device_vector<cuda_Board> dBoard{board};
    thrust::device_vector<int> dCount(1);

    GetGPUMovesKernel<<<1, 1>>>(thrust::raw_pointer_cast(dBoard.data()), thrust::raw_pointer_cast(dMoves.data()),
                                thrust::raw_pointer_cast(dCount.data()));
    const auto rc = cudaDeviceSynchronize();
    CUDA_TRACE_ERROR(rc);

    if (rc != cudaSuccess) {
        throw std::runtime_error("Failed to launch kernel");
    }

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

        if (hMove.GetPackedIndexes() != cMove[1]) {
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
}

void MoveGenTest_(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    std::cout << "MoveGen Test" << std::endl;

    std::cout << "Testing positions with depth 0" << std::endl;
    for (const auto &fen: TestFEN) {
        TestSinglePositionOutput(fen);
    }

    std::cout << "Testing positions with non zero depth" << std::endl;
    for (size_t i = 0; i < TestFEN.size(); ++i) {
        TestMoveCount(TestFEN[i], TestDepths[i]);
    }
}

std::vector<std::string> LoadFenDb() {
    const auto sourcePath = std::filesystem::path(__FILE__).parent_path();
    const auto fenDbPath = sourcePath / "test_data/fen_db.txt";

    const auto fenDb = cpu::LoadFenDb(fenDbPath);
    std::cout << "Loaded " << fenDb.size() << " positions" << std::endl;

    return fenDb;
}


std::vector<__uint64_t> GenSeeds(const __uint32_t size) {
    std::vector<__uint64_t> seeds{};
    seeds.reserve(size);

    std::mt19937_64 rng{std::random_device{}()};
    for (__uint32_t i = 0; i < size; ++i) {
        seeds.push_back(rng());
    }
    return seeds;
}

static constexpr size_t RETRIES = 3;
static constexpr size_t MAX_DEPTH = 100;

void MoveGenPerfGPU(__uint32_t blocks, __uint32_t threads, const std::vector<std::string> &fenDb,
                    const std::vector<__uint64_t> &seeds) {
    std::cout << "Running MoveGen Performance Test on GPU" << std::endl;
}

void MoveGenPerfTest(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    const auto fenDb = LoadFenDb();

    const auto [blocks, threads] = GetDims(threadsAvailable, deviceProps);
    const auto sizeThreads = blocks * threads;

    const auto seeds = GenSeeds(sizeThreads);

    if (fenDb.size() < threadsAvailable) {
        throw std::runtime_error("Not enough positions in the database");
    }

    std::cout << "MoveGen Performance Test" << std::endl;

    std::cout << std::string(80, '-') << std::endl;
    MoveGenPerfGPU(blocks, threads, fenDb, seeds);
    std::cout << std::string(80, '-') << std::endl;
    cpu::TestMoveGenPerfCPU(fenDb, MAX_DEPTH, RETRIES, sizeThreads, seeds);
}

void MoveGenTest(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
        MoveGenTest_(threadsAvailable, deviceProps);
        MoveGenPerfTest(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Failed test with Error: " << e.what() << std::endl;
    }
}
