//
// Created by Jlisowskyy on 21/11/24.
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
#include <chrono>
#include <format>

static constexpr size_t RETRIES = 3;
static constexpr size_t MAX_DEPTH = 100;

void __global__
SimulateGamesKernel(cuda_Board *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves, int maxDepth) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    __uint32_t seed = seeds[idx];

    int depth{};

    while (depth < maxDepth) {
        Stack<cuda_Move> stack(moves + idx * 256);
        MoveGenerator mGen{*(boards + idx), stack};
        const auto generatedMoves = mGen.GetMovesFast();
        __syncthreads();

        auto result = (__uint32_t *) (results + idx);
        ++result[0];
        result[1] += generatedMoves.size;

        if (generatedMoves.size == 0) {
            break;
        }

        const auto nextMove = generatedMoves[seed % generatedMoves.size];
        cuda_Move::MakeMove(nextMove, *(boards + idx));

        simpleRand(seed);
        ++depth;
    }
}

std::vector<std::string> LoadFenDb() {
    const auto sourcePath = std::filesystem::path(__FILE__).parent_path();
    const auto fenDbPath = sourcePath / "test_data/fen_db.txt";

    const auto fenDb = cpu::LoadFenDb(fenDbPath);
    std::cout << "Loaded " << fenDb.size() << " positions" << std::endl;

    return fenDb;
}

std::vector<__uint32_t> GenSeeds(const __uint32_t size) {
    std::vector<__uint32_t> seeds{};
    seeds.reserve(size);

    std::mt19937 rng{std::random_device{}()};
    for (__uint32_t i = 0; i < size; ++i) {
        seeds.push_back(rng());
    }
    return seeds;
}

void DisplayPerfResults(const double seconds, const __uint64_t boardEvaluated, __uint64_t movesGenerated) {
    const double milliseconds = seconds * 1000.0;
    const double boardsEvaluatedPerMs = static_cast<double>(boardEvaluated) / milliseconds;
    const double boardsEvaluatedPerS = boardsEvaluatedPerMs * 1000.0;
    const double movesGeneratedPerMs = static_cast<double>(movesGenerated) / milliseconds;
    const double movesGeneratedPerS = movesGeneratedPerMs * 1000.0;

    std::cout << std::format("Test on GPU took: {}s / {}ms\n"
                             "Reached board generation score: {} per ms, {} per s\n"
                             "Reached move generation score: {} per ms, {} per s\n"
                             "Total boards evaluated: {}\n"
                             "Total moves generated: {}",
                             seconds, milliseconds, boardsEvaluatedPerMs, boardsEvaluatedPerS, movesGeneratedPerMs,
                             movesGeneratedPerS, boardEvaluated, movesGenerated) << std::endl;
}

void DisplayPerfResults(const double seconds, const thrust::host_vector<__uint64_t> &results) {
    __uint64_t boardEvaluated{};
    __uint64_t movesGenerated{};
    for (const __uint64_t result: results) {
        const auto pScore = (const __uint32_t *) (&result);
        const __uint32_t bEval = pScore[0];
        const __uint32_t mGen = pScore[1];

        boardEvaluated += bEval;
        movesGenerated += mGen;
    }

    DisplayPerfResults(seconds, boardEvaluated, movesGenerated);
}

void MoveGenPerfGPU(__uint32_t blocks, __uint32_t threads, const std::vector<std::string> &fenDb,
                    const std::vector<__uint32_t> &seeds) {
    const __uint32_t threadsSize = threads * blocks;
    assert(threadsSize == seeds.size());

    std::cout << "Running MoveGen Performance Test on GPU" << std::endl;

    std::vector<cuda_Board> boards(threadsSize);

    for (__uint32_t i = 0; i < threadsSize; ++i) {
        boards[i] = cuda_Board(cpu::TranslateFromFen(fenDb[i]));
    }

    thrust::device_vector<__uint32_t> dSeeds = seeds;
    thrust::device_vector<__uint64_t> dResults(threadsSize);
    thrust::device_vector<cuda_Move> dMoves(threadsSize * 256);

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < RETRIES; ++i) {
        thrust::device_vector<cuda_Board> dBoards = boards;
        SimulateGamesKernel<<<4 * blocks, threads / 4>>>(thrust::raw_pointer_cast(dBoards.data()),
                                                         thrust::raw_pointer_cast(dSeeds.data()),
                                                         thrust::raw_pointer_cast(dResults.data()),
                                                         thrust::raw_pointer_cast((dMoves.data())), MAX_DEPTH);
        CUDA_TRACE_ERROR(cudaGetLastError());

        const auto rc = cudaDeviceSynchronize();
        CUDA_TRACE_ERROR(rc);

        if (rc != cudaSuccess) {
            throw std::runtime_error("Failed to launch kernel");
        }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    thrust::host_vector<__uint64_t> hResults = dResults;

    const double seconds = std::chrono::duration<double>(t2 - t1).count();
    DisplayPerfResults(seconds, hResults);
}

void MoveGenPerfTest_(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
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
    const auto [seconds, boardResults, moveResults] = cpu::TestMoveGenPerfCPU(fenDb, MAX_DEPTH, sizeThreads, RETRIES,
                                                                              seeds);
    DisplayPerfResults(seconds, boardResults, moveResults);
}

void MoveGenPerfTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
        MoveGenPerfTest_(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Failed test with Error: " << e.what() << std::endl;
    }
}