//
// Created by Jlisowskyy on 21/11/24.
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

static constexpr __uint32_t RETRIES = 2;
static constexpr __uint32_t MAX_DEPTH = 100;

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

template<class FuncT, __uint32_t BATCH_SIZE>
void SimpleTester(FuncT func, __uint32_t threadsAvailable, const cudaDeviceProp &deviceProps,
                  const std::vector<std::string> &fenDb,
                  const std::vector<__uint32_t> &seeds) {
    const __uint32_t blocks = threadsAvailable / BATCH_SIZE;
    const auto sizeThreads = blocks * BATCH_SIZE;
    assert(sizeThreads == threadsAvailable && "Wrongly generated block/threads");
    assert(sizeThreads == seeds.size());

    std::vector<cuda_Board> boards(sizeThreads);

    for (__uint32_t i = 0; i < sizeThreads; ++i) {
        boards[i] = cuda_Board(cpu::TranslateFromFen(fenDb[i]));
    }

    thrust::device_vector<__uint32_t> dSeeds = seeds;
    thrust::device_vector<__uint64_t> dResults(sizeThreads);
    thrust::device_vector<cuda_Move> dMoves(sizeThreads * 256);

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (__uint32_t i = 0; i < RETRIES; ++i) {
        thrust::device_vector<DefaultPackedBoardT> dBoards{DefaultPackedBoardT(boards)};
        func<<<blocks, BATCH_SIZE>>>(thrust::raw_pointer_cast(dBoards.data()),
                                     thrust::raw_pointer_cast(dSeeds.data()),
                                     thrust::raw_pointer_cast(dResults.data()),
                                     thrust::raw_pointer_cast((dMoves.data())), MAX_DEPTH);
        CUDA_TRACE_ERROR(cudaGetLastError());
    }
    GUARDED_SYNC();

    const auto t2 = std::chrono::high_resolution_clock::now();
    thrust::host_vector<__uint64_t> hResults = dResults;

    const double seconds = std::chrono::duration<double>(t2 - t1).count();
    DisplayPerfResults(seconds, hResults);
}

template<class FuncT>
void SplitTester(FuncT func, __uint32_t totalBoardsToProcess, const std::vector<std::string> &fenDb,
                 const std::vector<__uint32_t> &seeds) {
    assert((totalBoardsToProcess / SINGLE_RUN_BOARDS_SIZE) * SINGLE_RUN_BOARDS_SIZE == totalBoardsToProcess);
    std::vector<cuda_Board> boards(totalBoardsToProcess);

    for (__uint32_t i = 0; i < totalBoardsToProcess; ++i) {
        boards[i] = cuda_Board(cpu::TranslateFromFen(fenDb[i]));
    }

    thrust::device_vector<__uint32_t> dSeeds = seeds;
    thrust::device_vector<__uint64_t> dResults(totalBoardsToProcess);
    thrust::device_vector<cuda_Move> dMoves(totalBoardsToProcess * 256);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const __uint32_t bIdxRange = totalBoardsToProcess / SINGLE_RUN_BOARDS_SIZE;
    for (__uint32_t i = 0; i < RETRIES; ++i) {
        thrust::device_vector<DefaultPackedBoardT> dBoards{DefaultPackedBoardT(boards)};

        for (__uint32_t bIdx = 0; bIdx < bIdxRange;) {
            for (__uint32_t j = 0; j < 2 && bIdx < bIdxRange; ++j, ++bIdx) {
                func<<<SINGLE_RUN_BLOCK_SIZE, SINGLE_BATCH_SIZE>>>(
                        thrust::raw_pointer_cast(dBoards.data()) + bIdx * SINGLE_RUN_BOARDS_SIZE,
                        thrust::raw_pointer_cast(dSeeds.data()) + bIdx * SINGLE_RUN_BOARDS_SIZE,
                        thrust::raw_pointer_cast(dResults.data()) + bIdx * SINGLE_RUN_BOARDS_SIZE,
                        thrust::raw_pointer_cast((dMoves.data())) + bIdx * SINGLE_RUN_BOARDS_SIZE * 256, MAX_DEPTH);
                CUDA_TRACE_ERROR(cudaGetLastError());
            }
            GUARDED_SYNC();
        }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    thrust::host_vector<__uint64_t> hResults = dResults;

    const double seconds = std::chrono::duration<double>(t2 - t1).count();
    DisplayPerfResults(seconds, hResults);
}

void
MoveGenPerfGPUV1(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps, const std::vector<std::string> &fenDb,
                 const std::vector<__uint32_t> &seeds) {

    std::cout << "Running MoveGen V1 Performance Test on GPU" << std::endl;
    SimpleTester<decltype(SimulateGamesKernel), SINGLE_THREAD_SINGLE_GAME_BATCH_SIZE>(SimulateGamesKernel,
                                                                                      threadsAvailable, deviceProps,
                                                                                      fenDb, seeds);
}

void MoveGenPerfGPUV2(__uint32_t totalBoardsToProcess, const std::vector<std::string> &fenDb,
                      const std::vector<__uint32_t> &seeds) {
    \

    std::cout << "Running MoveGen V2 Performance Test on GPU" << std::endl;
    SplitTester(SimulateGamesKernelSplitMoves, totalBoardsToProcess, fenDb, seeds);
}

void MoveGenPerfGPUV4(__uint32_t totalBoardsToProcess, const std::vector<std::string> &fenDb,
                      const std::vector<__uint32_t> &seeds) {

    std::cout << "Running MoveGen V4 Performance Test on GPU" << std::endl;
    SplitTester(SimulateGamesKernelSplitMovesShared, totalBoardsToProcess, fenDb, seeds);
}

void
MoveGenPerfGPUV3(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps, const std::vector<std::string> &fenDb,
                 const std::vector<__uint32_t> &seeds) {

    std::cout << "Running MoveGen V3 Performance Test on GPU" << std::endl;
    SimpleTester<decltype(SimulateGamesKernelShared), SINGLE_THREAD_SINGLE_GAME_SHARED_BATCH_SIZE>(
            SimulateGamesKernelShared, threadsAvailable,
            deviceProps, fenDb, seeds);
}

void MoveGenPerfTest_(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    const auto fenDb = LoadFenDb();
    const auto seeds = GenSeeds(threadsAvailable);

    if (fenDb.size() < threadsAvailable) {
        throw std::runtime_error("Not enough positions in the database");
    }

    std::cout << "MoveGen Performance Test" << std::endl;

    std::cout << std::string(80, '-') << std::endl;
    MoveGenPerfGPUV1(threadsAvailable, deviceProps, fenDb, seeds);
    std::cout << std::string(80, '-') << std::endl;
    MoveGenPerfGPUV2(threadsAvailable, fenDb, seeds);
    std::cout << std::string(80, '-') << std::endl;
//    MoveGenPerfGPUV3(threadsAvailable, deviceProps, fenDb, seeds);
    std::cout << std::string(80, '-') << std::endl;
//    MoveGenPerfGPUV4(threadsAvailable, fenDb, seeds);
    std::cout << std::string(80, '-') << std::endl;
    const auto [seconds, boardResults, moveResults] = cpu::TestMoveGenPerfCPU(fenDb, MAX_DEPTH, threadsAvailable,
                                                                              RETRIES,
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