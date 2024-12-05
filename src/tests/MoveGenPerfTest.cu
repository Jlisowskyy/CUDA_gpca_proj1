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
#include <random>
#include <chrono>
#include <format>

static constexpr __uint32_t RETRIES = 2;
static constexpr __uint32_t MAX_DEPTH = 100;

/**
 * @brief Displays performance results of move generation tests.
 *
 * @param seconds Total test execution time in seconds
 * @param boardEvaluated Total number of chess boards processed
 * @param movesGenerated Total number of moves generated
 */
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

/**
 * @brief Overloaded performance results display for vector-based results.
 *
 * Aggregates performance metrics from a vector of 64-bit result entries,
 * where each entry contains board and move generation statistics.
 *
 * @param seconds Total test execution time in seconds
 * @param results Vector of performance result entries
 */
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

/**
 * @brief Performs a templated performance test for simple move generation scenarios (1 thread 1 board)
 *
 * Conducts a standardized performance test with configurable batch size and
 * move generation kernel. Handles board setup, device memory allocation,
 * and performance measurement.
 *
 * @tparam FuncT Type of the move generation kernel function
 * @tparam BATCH_SIZE Number of threads processed in a single batch
 *
 * @param func Move generation kernel to be tested
 * @param threadsAvailable Total number of CUDA threads for testing
 * @param deviceProps CUDA device properties
 * @param fenDb Database of chess positions in FEN notation
 * @param seeds Random seeds for test reproducibility
 *
 * @note Performs multiple test retries to ensure consistent results
 */
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

    auto *packedBoard = new DefaultPackedBoardT(boards);

    DefaultPackedBoardT *d_boards;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_boards, sizeof(DefaultPackedBoardT)));

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (__uint32_t i = 0; i < RETRIES; ++i) {
        CUDA_ASSERT_SUCCESS(cudaMemcpy(d_boards, packedBoard, sizeof(DefaultPackedBoardT), cudaMemcpyHostToDevice));

        func<<<blocks, BATCH_SIZE>>>(d_boards,
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

    CUDA_ASSERT_SUCCESS(cudaFree(d_boards));
    delete packedBoard;
}

/**
 * @brief Conducts a performance test with split move generation approach.
 *
 * Tests move generation using a more complex kernel that processes boards
 * in smaller, split batches to potentially improve GPU utilization.
 *
 * @tparam FuncT Type of the split move generation kernel function
 *
 * @param func Split move generation kernel to be tested
 * @param totalBoardsToProcess Total number of chess boards to process
 * @param fenDb Database of chess positions in FEN notation
 * @param seeds Random seeds for test reproducibility
 *
 * @note Designed to handle larger numbers of boards with more granular processing
 */
template<class FuncT>
void SplitTester(FuncT func, __uint32_t totalBoardsToProcess, const std::vector<std::string> &fenDb,
                 const std::vector<__uint32_t> &seeds) {
    std::vector<cuda_Board> boards(totalBoardsToProcess);

    for (__uint32_t i = 0; i < totalBoardsToProcess; ++i) {
        boards[i] = cuda_Board(cpu::TranslateFromFen(fenDb[i]));
    }

    thrust::device_vector<__uint32_t> dSeeds = seeds;
    thrust::device_vector<__uint64_t> dResults(totalBoardsToProcess);
    auto *packedBoard = new DefaultPackedBoardT(boards);

    DefaultPackedBoardT *d_boards;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_boards, sizeof(DefaultPackedBoardT)));

    const __uint32_t bIdxRange = totalBoardsToProcess / SINGLE_RUN_BOARDS_SIZE;

    const auto t1 = std::chrono::high_resolution_clock::now();
    for (__uint32_t i = 0; i < RETRIES; ++i) {
        CUDA_ASSERT_SUCCESS(cudaMemcpy(d_boards, packedBoard, sizeof(DefaultPackedBoardT), cudaMemcpyHostToDevice));

        for (__uint32_t bIdx = 0; bIdx < bIdxRange;) {
            for (__uint32_t j = 0; j < 4 && bIdx < bIdxRange; ++j, ++bIdx) {
                func<<<SINGLE_RUN_BLOCK_SIZE, SPLIT_BATCH_NUM_THREADS>>>(
                        d_boards,
                        thrust::raw_pointer_cast(dSeeds.data()) + bIdx * SINGLE_RUN_BOARDS_SIZE,
                        thrust::raw_pointer_cast(dResults.data()) + bIdx * SINGLE_RUN_BOARDS_SIZE, MAX_DEPTH);
                CUDA_TRACE_ERROR(cudaGetLastError());
            }
            GUARDED_SYNC();
        }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    thrust::host_vector<__uint64_t> hResults = dResults;

    const double seconds = std::chrono::duration<double>(t2 - t1).count();
    DisplayPerfResults(seconds, hResults);
    CUDA_ASSERT_SUCCESS(cudaFree(d_boards));
    delete packedBoard;
}

/**
 * @brief Performs move generation performance test using Version 1 approach.
 *
 * Executes a GPU-based move generation performance test with single-thread,
 * single-game batch processing strategy.
 *
 * @param threadsAvailable Total number of CUDA threads for testing
 * @param deviceProps CUDA device properties
 * @param fenDb Database of chess positions in FEN notation
 * @param seeds Random seeds for test reproducibility
 */
void
MoveGenPerfGPUV1(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps, const std::vector<std::string> &fenDb,
                 const std::vector<__uint32_t> &seeds) {

    std::cout << "Running MoveGen V1 Performance Test on GPU" << std::endl;
    SimpleTester<decltype(SimulateGamesKernel), SINGLE_THREAD_SINGLE_GAME_BATCH_SIZE>(SimulateGamesKernel,
                                                                                      threadsAvailable, deviceProps,
                                                                                      fenDb, seeds);
}

/**
 * @brief Performs move generation performance test using Version 2 approach.
 *
 * Executes a GPU-based move generation performance test with split moves
 * processing strategy, potentially improving GPU resource utilization.
 *
 * @param totalBoardsToProcess Total number of chess boards to process
 * @param fenDb Database of chess positions in FEN notation
 * @param seeds Random seeds for test reproducibility
 */
void MoveGenPerfGPUV2(__uint32_t totalBoardsToProcess, const std::vector<std::string> &fenDb,
                      const std::vector<__uint32_t> &seeds) {
    std::cout << "Running MoveGen V2 Performance Test on GPU" << std::endl;
    SplitTester(SimulateGamesKernelSplitMoves, totalBoardsToProcess, fenDb, seeds);
}

/**
 * @brief Diagnostic function to test and display split index calculations.
 *
 * Generates and prints split index information for a predefined number of
 * blocks and threads, helping to verify the indexing strategy for split
 * move generation.
 *
 * @note Uses static constexpr values for blocks and block size
 */
void TestSplitIndexes() {
    static constexpr __uint32_t BLOCKS = 2;
    static constexpr __uint32_t BLOCK_SIZE = 384;

    std::cout << "Testing split indexes:" << std::endl;
    for (__uint32_t bx = 0; bx < BLOCKS; ++bx) {
        for (__uint32_t tx = 0; tx < BLOCK_SIZE; ++tx) {
            const auto [plainIdx, boardIdx, figIdx, counterIdx] = CalcSplitIdx(tx, bx, BLOCK_SIZE);

            std::cout << std::format("[ bx: {}, tx: {} ] : [ plainIdx: {}, boardIdx: {}, figIdx: {}, counterIdx: {}]",
                                     bx, tx, plainIdx, boardIdx, figIdx, counterIdx
            ) << std::endl;
        }
    }
}

/**
 * @brief Internal implementation of move generation performance testing.
 *
 * Comprehensive performance test that includes:
 * 1. Loading chess position database
 * 2. Generating random seeds
 * 3. Executing GPU Version 1 performance test
 * 4. Executing GPU Version 2 performance test
 * 5. Running CPU reference performance test
 *
 * @param threadsAvailable Total number of CUDA threads for testing
 * @param deviceProps CUDA device properties
 *
 * @throws std::runtime_error If insufficient chess positions are available
 */
void MoveGenPerfTest_(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    const auto fenDb = LoadFenDb();
    const auto seeds = GenSeeds(threadsAvailable);

    if (fenDb.size() < threadsAvailable) {
        throw std::runtime_error("Not enough positions in the database");
    }

    std::cout << "MoveGen Performance Test" << std::endl;

//    TestSplitIndexes();

    std::cout << std::string(80, '-') << std::endl;
    MoveGenPerfGPUV1(threadsAvailable, deviceProps, fenDb, seeds);
    PolluteCache();
    std::cout << std::string(80, '-') << std::endl;
    MoveGenPerfGPUV2(threadsAvailable, fenDb, seeds);
    PolluteCache();
    std::cout << std::string(80, '-') << std::endl;
    const auto [seconds, boardResults, moveResults] = cpu::TestMoveGenPerfCPU(fenDb, MAX_DEPTH, threadsAvailable,
                                                                              RETRIES,
                                                                              seeds);
    DisplayPerfResults(seconds, boardResults, moveResults);
}

/**
 * @brief Wrapper for move generation performance testing with error handling.
 *
 * Calls the internal performance test implementation and catches any
 * exceptions that might occur during testing, ensuring robust error reporting.
 *
 * @param threadsAvailable Total number of CUDA threads for testing
 * @param deviceProps CUDA device properties
 */
void MoveGenPerfTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
        MoveGenPerfTest_(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Failed test with Error: " << e.what() << std::endl;
    }
}