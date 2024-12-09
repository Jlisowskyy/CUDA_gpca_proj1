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
#include "../cpu_core/ThreadPool.cuh"

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

/* This position will always be evaluated as first shot during each test run to simplify position debugging */
static constexpr const char *MAIN_TEST_FEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

/**
 * @brief Defines the recursion depth for move generation testing
 *
 * Controls how long move sequences will be explored during
 * comprehensive move generation validation.
 */
static constexpr uint32_t TEST_DEPTH = 4;

/**
 * @brief Macro for flexible message logging during testing
 *
 * Provides a unified logging mechanism that can output messages:
 * - To a progress bar if available
 * - Directly to standard output if no progress bar exists
 *
 * Enables consistent, conditional logging across different
 * testing scenarios and environments.
 *
 * @param msg The message to be logged
 */
#define DUMP_MSG(msg)                     \
    if (progBar != nullptr) {             \
        progBar->WriteLine(msg);          \
    } else {                              \
        std::cout << msg << std::endl;    \
    }                                     \


/**
 * @brief Compile-time flag to control verbose output during testing
 *
 * When set to true, enables detailed logging and output
 * for move generation tests. Controlled via compile-time
 * definition to allow for zero-overhead logging control.
 *
 * - If defined with WRITE_OUT: Verbose logging is enabled
 * - If not defined: Minimal logging is used
 */
#ifdef WRITE_OUT

static constexpr bool WRITE_OUT = true;

#else

static constexpr bool WRITE_OUT = false;

#endif

/* THREAD POOL SIZE FOR TESTS */
static constexpr uint32_t NUM_TEST_THREADS = 64;

// ------------------------------
// Test kernels
// ------------------------------


/**
 * @brief CUDA kernel function to generate moves for a chess board on the GPU using a standard move generation approach
 *
 * @param board Pointer to a packed representation of the chess board
 * @param outMoves Output array to store generated moves
 * @param outCount Pointer to store the total number of moves generated
 */
void __global__ GetGPUMovesKernel(SmallPackedBoardT *board, cuda_Move *outMoves, int *outCount) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ stub sharedStack[DEFAULT_STACK_SIZE];
    __shared__ uint32_t counter;
    Stack<cuda_Move> stack((cuda_Move*)sharedStack, &counter);

    MoveGenerator<1> gen{(*board)[0], stack};

    gen.GetMovesFast();
    *outCount = static_cast<int>(stack.Size());

    for (int i = 0; i < stack.Size(); ++i) {
        outMoves[i] = stack[i];
    }
}

/**
 * @brief CUDA kernel function to generate moves using a split move generation strategy
 *
 * @param board Pointer to a packed representation of the chess board
 * @param outMoves Output array to store generated moves
 * @param outCount Pointer to store the total number of moves generated
 *
 * @note This kernel generates moves by distributing the work across multiple threads,
 * with each thread responsible for generating moves for a specific piece type.
 */
void __global__ GetGPUMovesKernelSplit(SmallPackedBoardT *board, cuda_Move *outMoves, int *outCount) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    const uint64_t figIdx = threadIdx.x % 6;
    __shared__ stub sharedStack[DEFAULT_STACK_SIZE];
    __shared__ uint32_t counter;
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

/**
 * @brief CUDA kernel to recursively count the total number of possible moves at a given depth
 *
 * @param board Pointer to a packed representation of the chess board
 * @param depth Maximum depth of move tree to explore
 * @param outCount Pointer to store the total number of moves counted
 *
 * @note Generates a complete move count by exploring all possible move sequences
 * up to the specified depth on the GPU. Utilizes basic simple algorithm.
 */
void __global__ GetGPUMoveCountsKernel(SmallPackedBoardT *board, const int depth, uint64_t *outCount) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ stub sharedStack[10][DEFAULT_STACK_SIZE];
    Stack<cuda_Move> stack(sharedStack);
    MoveGenerator<1> gen{(*board)[0], stack};

    const auto moves = gen.CountMoves(depth, sharedStack);
    *outCount = moves;
}

/**
 * @brief CUDA kernel to count moves using a split move generation strategy
 *
 * @param board Pointer to a packed representation of the chess board
 * @param depth Maximum depth of move tree to explore
 * @param outCount Pointer to store the total number of moves counted
 *
 * @note Similar to GetGPUMoveCountsKernel, but distributes move counting across
 * multiple threads, with each thread handling moves for a specific piece type.
 */
void __global__ GetGPUMoveCountsKernelSplit(SmallPackedBoardT *board, const int depth, uint64_t *outCount) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ stub sharedStack[10][DEFAULT_STACK_SIZE];
    Stack<cuda_Move> stack(sharedStack, false);
    MoveGenerator<1> gen{(*board)[0], stack};

    /* NOTE: threadIdx < 6 */
    const uint32_t figIdx = threadIdx.x;

    if (figIdx == 0) {
        stack.Clear();
    }

    __syncthreads();
    const auto moves = gen.CountMovesSplit(figIdx, depth, sharedStack);
    __syncthreads();

    if (figIdx == 0) {
        *outCount = moves;
    }
}

// ------------------------------
// Stream wrappers
// ------------------------------

void RunBaseKernel(SmallPackedBoardT *dBoard, cuda_Move *dMoves, int *dCount, cudaStream_t *stream = nullptr) {
    if (stream) {
        GetGPUMovesKernel<<<1, 1, 0, *stream>>>(dBoard, dMoves, dCount);
    } else {
        GetGPUMovesKernel<<<1, 1>>>(dBoard, dMoves, dCount);
    }
};

void RunSplitKernel(SmallPackedBoardT *dBoard, cuda_Move *dMoves, int *dCount, cudaStream_t *stream = nullptr) {
    if (stream) {
        GetGPUMovesKernelSplit<<<1, 6, 0, *stream>>>(dBoard, dMoves, dCount);
    } else {
        GetGPUMovesKernelSplit<<<1, 6>>>(dBoard, dMoves, dCount);
    }
};

void RunBaseKernelMoveCount(SmallPackedBoardT *boards, const int depth, uint64_t *outCount,
                            cudaStream_t *stream = nullptr) {
    if (stream) {
        GetGPUMoveCountsKernel<<<1, 1, 0, *stream>>>(boards, depth, outCount);
    } else {
        GetGPUMoveCountsKernel<<<1, 1>>>(boards, depth, outCount);
    }
}

void RunSplitKernelMoveCount(SmallPackedBoardT *boards, const int depth, uint64_t *outCount,
                             cudaStream_t *stream = nullptr) {
    if (stream) {
        GetGPUMoveCountsKernelSplit<<<1, 6, 0, *stream>>>(boards, depth, outCount);
    } else {
        GetGPUMoveCountsKernelSplit<<<1, 1>>>(boards, depth, outCount);
    }
}

// ------------------------------
// Utilities
// ------------------------------

/**
 * @brief Template function to generate chess moves using a specified CUDA kernel
 *
 * @tparam FuncT Type of the CUDA kernel function
 * @param func The CUDA kernel function to use for move generation
 * @param board The chess board to generate moves for
 * @return std::vector<cuda_Move> Vector of generated moves
 *
 * @note Manages CUDA memory allocation, kernel execution, and move transfer
 * between device and host memory. Provides a high-level interface
 * for GPU-based move generation. Designed to be used together with many CPU threads.
 */
template<class FuncT>
std::vector<cuda_Move> GenerateMovesByKernel(FuncT func, const cuda_Board &board) {
    cudaStream_t stream;
    CUDA_ASSERT_SUCCESS(cudaStreamCreate(&stream));

    auto packedBoard = new SmallPackedBoardT(std::vector{board});
    SmallPackedBoardT *dBoard{};
    cuda_Move *dMoves{};
    int *dCount{};

    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dBoard, sizeof(SmallPackedBoardT), stream));
    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dMoves, sizeof(cuda_Move) * DEFAULT_STACK_SIZE, stream));
    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dCount, sizeof(int), stream));
    CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dBoard, packedBoard,
                                        sizeof(SmallPackedBoardT), cudaMemcpyHostToDevice, stream));
    func(dBoard, dMoves, dCount, &stream);

    std::vector<cuda_Move> cudaMoves(DEFAULT_STACK_SIZE);
    int hCount{};

    CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));

    CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(cudaMoves.data(), dMoves,
                                        sizeof(cuda_Move) * DEFAULT_STACK_SIZE, cudaMemcpyDeviceToHost, stream));
    CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(&hCount, dCount,
                                        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));

    CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBoard, stream));
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(dMoves, stream));
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(dCount, stream));

    cudaMoves.resize(hCount);

    delete packedBoard;

    CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));
    CUDA_ASSERT_SUCCESS(cudaStreamDestroy(stream));

    return cudaMoves;
}

/**
 * @brief Compares move counts between CPU and GPU implementations
 *
 * @tparam FuncT Type of the CUDA move counting kernel
 * @param func The move counting kernel function
 * @param fen FEN string representing the chess board position
 * @param depth Depth of move tree to explore
 * @param progBar Optional progress bar for testing
 * @param writeToOut Flag to enable verbose output
 * @return bool Indicates whether CPU and GPU move counts match
 *
 * @note Performs a comprehensive test comparing move counts generated by
 * CPU and GPU implementations to validate GPU move generation accuracy.
 */
template<class FuncT>
bool TestMoveCount(FuncT func,
                   const std::string_view &fen,
                   int depth,
                   ProgressBar *progBar = nullptr,
                   bool writeToOut = false) {
    if (writeToOut) {
        DUMP_MSG(std::format("Testing position: {}", fen))
    }

    const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
    cuda_Board board(externalBoard);

    thrust::device_vector<uint64_t> dCount(1);
    thrust::device_vector<cuda_Move> dStack(16384);

    auto *packedBoard = new SmallPackedBoardT(std::vector{board});
    SmallPackedBoardT *dBoard;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&dBoard, sizeof(SmallPackedBoardT)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(dBoard, packedBoard, sizeof(SmallPackedBoardT), cudaMemcpyHostToDevice));

    func(dBoard, depth, thrust::raw_pointer_cast(dCount.data()), nullptr);
    CUDA_TRACE_ERROR(cudaGetLastError());
    GUARDED_SYNC();

    thrust::host_vector<uint64_t> hdCount = dCount;
    const uint64_t hCount = hdCount[0];

    const auto cCount = cpu::CountMoves(externalBoard, depth);

    CUDA_ASSERT_SUCCESS(cudaFree(dBoard));
    delete packedBoard;

    if (cCount != hCount) {
        const std::string msg = std::format("Moves count mismatch: cpu {} !=  gpu {}\nPosition: {}",
                                            cCount,
                                            hCount,
                                            fen
        );
        DUMP_MSG(msg)
        return false;
    } else {
        if (writeToOut) {
            DUMP_MSG(std::format("Test passed for: {}", fen))
        }
        return true;
    }
}

/**
 * @brief Validates moves generated by GPU against CPU-generated moves
 *
 * @param fen FEN string of the chess board position
 * @param cMoves Moves generated by CPU
 * @param hMoves Moves generated by GPU
 * @param progBar Optional progress bar for testing
 * @param writeToOut Flag to enable verbose output
 * @return std::string Description of any move generation discrepancies
 *
 * @details Performs a detailed comparison of moves, checking:
 * - Total move count
 * - Individual move correctness
 * - Move representation consistency
 * Returns an empty string if moves match, otherwise returns error details.
 */
std::string ValidateMoves(const std::string &fen,
                   const std::vector<cpu::external_move> &cMoves,
                   const std::vector<cuda_Move> &hMoves,
                   ProgressBar *progBar = nullptr,
                   bool writeToOut = false) {

    if (cMoves.size() != hMoves.size()) {
        const auto msg = std::format("GPU move gen failed: Moves count mismatch: {} != {}", cMoves.size(),
                                     hMoves.size());
        DUMP_MSG(msg)
    }

    std::map<std::string, cuda_Move> GPUMovesMap{};
    for (const auto move: hMoves) {
        const auto [_, result] = GPUMovesMap.emplace(move.GetPackedMove().GetLongAlgebraicNotationCPU(), move);

        assert(result && "GPU move gen failed: received repeated move");
    }

    uint32_t errors{};
    std::string invalidMove{};
    for (const auto CPUmove: cMoves) {
        cuda_PackedMove convertedCPUMove{CPUmove[0]};

        const auto it = GPUMovesMap.find(convertedCPUMove.GetLongAlgebraicNotationCPU());

        if (it == GPUMovesMap.end()) {
            const auto msg = std::format("GPU move gen failed: {} was not generated by GPU",
                                         convertedCPUMove.GetLongAlgebraicNotationCPU());
            DUMP_MSG(msg)

            ++errors;

            invalidMove = "Lacking move: " + convertedCPUMove.GetLongAlgebraicNotationCPU();
            continue;
        }

        const cuda_Move gpuMove = it->second;

        if (gpuMove.GetPackedMove() != convertedCPUMove) {
            const auto msg = std::format("GPU move gen failed: Packed move mismatch device: {} != host {}",
                                         gpuMove.GetPackedMove().GetLongAlgebraicNotationCPU(),
                                         convertedCPUMove.GetLongAlgebraicNotationCPU());
            DUMP_MSG(msg)

            errors++;
        }

        /* Prohibit unused bits */
        static constexpr uint32_t CheckTypeBit = (uint32_t) 1 << 15;
        static constexpr uint32_t PackedIndexesMask = ~(CheckTypeBit);

        if (gpuMove.GetPackedIndexes() != (PackedIndexesMask & CPUmove[1])) {
            const auto msg = std::format(
                    "GPU move gen failed: Indexes mismatch device: {} != host {}\nDevice: {}\nHost: {}",
                    u16d(gpuMove.GetPackedIndexes()).to_string(),
                    u16d(CPUmove[1]).to_string(),
                    gpuMove.GetPackedMove().GetLongAlgebraicNotationCPU(),
                    convertedCPUMove.GetLongAlgebraicNotationCPU());
            DUMP_MSG(msg)

            invalidMove = convertedCPUMove.GetLongAlgebraicNotationCPU();
            errors++;
        }

        if (gpuMove.GetPackedMisc() != CPUmove[2]) {
            const auto msg = std::format(
                    "GPU move gen failed: Misc mismatch device: {} != host: {}\nDevice: {}\nHost: {}",
                    u16d(gpuMove.GetPackedMisc()).to_string(),
                    u16d(CPUmove[2]).to_string(),
                    gpuMove.GetPackedMove().GetLongAlgebraicNotationCPU(),
                    convertedCPUMove.GetLongAlgebraicNotationCPU()
            );

            DUMP_MSG(msg)

            invalidMove = convertedCPUMove.GetLongAlgebraicNotationCPU();
            errors++;
        }

        GPUMovesMap.erase(convertedCPUMove.GetLongAlgebraicNotationCPU());
    }

    for (const auto &[move, _]: GPUMovesMap) {
        const auto msg = std::format("GPU move gen failed: generated additional move: {}", move);

        DUMP_MSG(msg)

        invalidMove = "Additional move: " + move;
    }


    if (errors != 0) {
        std::string msg{};
        msg += "Failed test for position: " + std::string(fen) + '\n';
        msg += "Errors: " + std::to_string(errors) + '\n';
        msg += "Device moves:\n";

        int count{};
        for (const auto &move: hMoves) {
            msg += move.GetPackedMove().GetLongAlgebraicNotationCPU() + ", ";
            if (++count % 10 == 0) msg += '\n';
        }
        msg += '\n';

        msg += "Host moves:\n";
        count = 0;
        for (const auto &move: cMoves) {
            msg += cuda_PackedMove(move[0]).GetLongAlgebraicNotationCPU() + ", ";
            if (++count % 10 == 0) msg += '\n';
        }
        msg += '\n';

        DUMP_MSG(msg)
    } else {
        const auto msg = std::format("Test passed for position: {}", fen);

        if (writeToOut) {
            DUMP_MSG(msg)
        }
    }

    return invalidMove;
}

/**
 * @brief Recursively investigates move generation errors at different board states
 *
 * @tparam FuncT Type of move generation kernel
 * @param funcGen Move generation kernel function
 * @param board Current chess board state
 * @param depth Remaining depth to explore
 * @param progBar Optional progress bar for testing
 * @param writeToOut Flag to enable verbose output
 * @return std::string Detailed path of the first detected move generation error
 *
 * @note Explores move sequences to pinpoint specific board states
 * where GPU and CPU move generation diverge.
 */
template<class FuncT>
std::string FindWrongPathRecursive(FuncT funcGen,
                                   cuda_Board &board,
                                   int depth,
                                   ProgressBar *progBar = nullptr,
                                   bool writeToOut = false) {

    const auto cudaMoves = GenerateMovesByKernel(funcGen, board);
    const auto cpuMoves = cpu::GenerateMoves(board.DumpToExternal());

    if (std::string result = ValidateMoves(cpu::TranslateToFen(board.DumpToExternal()), cpuMoves, cudaMoves, progBar,
                                           writeToOut); !result.empty()) {
        return result + std::format(" Failed on depth: {}", depth);
    }

    if (depth == 1) {
        return "";
    }

    for (const auto CPUMove: cpuMoves) {
        cuda_Move move(CPUMove);

        VolatileBoardData data(board);
        cuda_Move::MakeMove(move, board);

        if (std::string result = FindWrongPathRecursive(funcGen, board, depth - 1, progBar,
                                                        writeToOut); !result.empty()) {
            return move.GetPackedMove().GetLongAlgebraicNotationCPU() + " --> " + result;
        }

        cuda_Move::UnmakeMove(move, board, data);
    }

    return "";
}

// ------------------------------
// Test runners
// ------------------------------

/**
 * @brief Validates move generation for a single chess board position
 *
 * @tparam FuncT Type of move generation kernel function
 * @param func Move generation kernel to test
 * @param fen FEN string representing the chess board position
 * @param progBar Optional progress bar for tracking test progress
 * @param writeToOut Flag to enable verbose output logging
 * @return std::string Detailed error message if move generation fails,
 *                     empty string if successful
 *
 * Comprehensive single-position move generation test that:
 * 1. Translates FEN to board representation
 * 2. Generates moves using both GPU and CPU methods
 * 3. Validates generated moves for:
 *    - Matching move count
 *    - Identical move representation
 *    - Consistency across different generation methods
 */
template<class FuncT>
std::string TestSinglePositionOutput(FuncT func, const std::string &fen, ProgressBar *progBar = nullptr,
                              bool writeToOut = false) {

    /* welcome message */
    if (writeToOut) {
        const auto msg = std::format("Testing moves of position: {}", fen);
        DUMP_MSG(msg)
    }

    /* Load boards */
    const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
    const cuda_Board board(externalBoard);

    const auto cudaMoves = GenerateMovesByKernel(func, board);
    const auto cMoves = cpu::GenerateMoves(externalBoard);

    /* validate returned moves */
    return ValidateMoves(std::string(fen), cMoves, cudaMoves, progBar, writeToOut);
}

/**
 * @brief Runs comprehensive move generation tests across multiple chess positions
 *
 * @tparam FuncT Type of move generation kernel
 * @param func Move generation kernel to test
 *
 * Executes move generation tests on:
 * 1. Predefined specialized test positions
 * 2. A large database of chess positions using multi-threaded testing
 *
 * Provides thorough validation of GPU move generation across diverse scenarios.
 */
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

    ThreadPool threadPool(NUM_TEST_THREADS);

    progBar.WriteLine("Testing move gen with whole fen db...");

    threadPool.RunThreads(
            [&](const uint32_t tid) {
                for (uint32_t idx = tid; idx < fens.size(); idx += NUM_TEST_THREADS) {
                    TestSinglePositionOutput(func, fens[idx], &progBar, WRITE_OUT);
                    progBar.Increment();
                }
            }
    );

    threadPool.Wait();
    std::cout << "TEST FINISHED" << std::endl;
}

/**
 * @brief Conducts comprehensive move generation testing across multiple depths
 *
 * @tparam FuncT Type of move counting kernel function
 * @tparam FuncT1 Type of move generation kernel function
 * @param funcCount Kernel function for counting moves
 * @param funcGen Kernel function for generating moves
 *
 * Systematically tests move generation by:
 * 1. Iterating through predefined test chess positions
 * 2. Verifying move counts match between CPU and GPU
 * 3. When move count verification fails, attempting to:
 *    - Identify the specific problematic move sequence
 *    - Locate the exact board state causing the discrepancy
 *
 * Provides a robust mechanism for validating CUDA move generation
 * across different board configurations and move depths.
 */
template<class FuncT, class FuncT1>
void RunDepthPosTest(FuncT funcCount, FuncT1 funcGen) {
    std::cout << "Testing move gen with specialized positions on " << TEST_DEPTH << " depth..." << std::endl;

    ProgressBar progBar(TestFEN.size(), 100);
    progBar.Start();

    for (auto fen: TestFEN) {
        if (!TestMoveCount(funcCount, fen, TEST_DEPTH, &progBar, WRITE_OUT)) {
            progBar.WriteLine(std::string(80, '-'));

            const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
            cuda_Board board(externalBoard);

            const std::string result = FindWrongPathRecursive(funcGen, board, TEST_DEPTH, &progBar, WRITE_OUT);

            if (result.empty()) {
                const std::string msg = std::format("Failed to find malfunctioning move on fen: {}", fen);
                progBar.WriteLine(msg);
            } else {
                const std::string msg = std::format("Found malfunctioning path: {}", result);
                progBar.WriteLine(msg);
            }

            progBar.WriteLine(std::string(80, '-'));
        }

        progBar.Increment();
    }

    std::cout << "TEST FINISHED" << std::endl;
}

/**
 * @brief Internal implementation of comprehensive CUDA move generation testing
 *
 * @param threadsAvailable Number of available CUDA threads (unused)
 * @param deviceProps CUDA device properties (unused)
 *
 * Performs an extensive validation of CUDA move generation by:
 * 1. Performing an initial critical test on a main test position
 * 2. Running tests for two move generation strategies:
 *    - Plain move generation
 *    - Split move generation
 * 3. Executing both single-position and depth-based move generation tests
 *
 * @note The function uses an underscore suffix to differentiate
 *       the implementation from the error-handling wrapper function
 *
 * @throws std::exception If critical testing fails
 */
void MoveGenTest_([[maybe_unused]] uint32_t threadsAvailable, [[maybe_unused]] const cudaDeviceProp &deviceProps) {
    static constexpr uint32_t EXTENDED_THREAD_STACK_SIZE = 16384;
    static constexpr uint32_t DEFAULT_THREAD_STACK_SIZE = 1024;

    std::cout << "MoveGen Test" << std::endl;

    std::cout << "Extending thread stack size to " << EXTENDED_THREAD_STACK_SIZE << "..." << std::endl;
    cudaDeviceSetLimit(cudaLimitStackSize, EXTENDED_THREAD_STACK_SIZE);

    const std::string result = TestSinglePositionOutput(RunBaseKernel, MAIN_TEST_FEN, nullptr, true);

    if (!result.empty()) {
        std::cout << "[ CRITICAL ERROR ] FIRST POSITION CHECK FAILED: " + result  << std::endl;
        exit(1);
    }

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Testing plain move gen: " << std::endl;

    RunSinglePosTest(RunBaseKernel);
    std::cout << std::string(80, '-') << std::endl;
    RunDepthPosTest(RunBaseKernelMoveCount, RunBaseKernel);

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Testing split move gen: " << std::endl;

    RunSinglePosTest(RunSplitKernel);
    std::cout << std::string(80, '-') << std::endl;
    RunDepthPosTest(RunSplitKernelMoveCount, RunSplitKernel);

    std::cout << std::string(80, '=') << std::endl;


    std::cout << "Reverting thread stack size to " << DEFAULT_THREAD_STACK_SIZE << "..." << std::endl;
    cudaDeviceSetLimit(cudaLimitStackSize, DEFAULT_THREAD_STACK_SIZE);
}

/**
 * @brief Primary test runner for CUDA chess move generation
 *
 * @param threadsAvailable Number of CUDA threads available
 * @param deviceProps CUDA device properties
 *
 * Orchestrates comprehensive testing of:
 * - Standard move generation kernel
 * - Split move generation kernel
 *
 * Includes setup for extended thread stack size and detailed error tracking.
 */
void MoveGenTest(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
        MoveGenTest_(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Failed test with Error: " << e.what() << std::endl;
    }
}
