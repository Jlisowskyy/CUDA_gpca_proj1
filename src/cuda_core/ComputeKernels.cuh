//
// Created by Jlisowskyy on 28/11/24.
//

#ifndef SRC_COMPUTEKERNELS_CUH
#define SRC_COMPUTEKERNELS_CUH

#include "Helpers.cuh"
#include "cuda_Board.cuh"
#include "Move.cuh"
#include "MoveGenerator.cuh"
#include "cuda_Board.cuh"
#include "cuda_PackedBoard.cuh"

#include <thrust/tuple.h>



static constexpr __uint32_t SINGLE_THREAD_SINGLE_GAME_BATCH_SIZE = 384;
static constexpr __uint32_t SINGLE_THREAD_SINGLE_GAME_STACK_SIZE = DEFAULT_STACK_SIZE;

static constexpr __uint32_t WARP_SIZE = 32;
static constexpr __uint32_t MINIMAL_SPLIT_BATCH_THREAD_SIZE = WARP_SIZE * BIT_BOARDS_PER_COLOR;
static constexpr __uint32_t SPLIT_BATCH_NUM_MINIMALS = 2;
static constexpr __uint32_t SPLIT_BATCH_NUM_THREADS = MINIMAL_SPLIT_BATCH_THREAD_SIZE * SPLIT_BATCH_NUM_MINIMALS;
static constexpr __uint32_t SINGLE_BATCH_BOARD_SIZE = WARP_SIZE * SPLIT_BATCH_NUM_MINIMALS;
static constexpr __uint32_t SINGLE_RUN_BLOCK_SIZE = 100;

static constexpr __uint32_t SINGLE_RUN_BOARDS_SIZE = SINGLE_RUN_BLOCK_SIZE * SINGLE_BATCH_BOARD_SIZE;
static constexpr __uint32_t SPLIT_MAX_STACK_MOVES = 80;

static constexpr __uint32_t NUM_ROUNDS_IN_MATERIAL_ADVANTAGE_TO_WIN = 5;
static constexpr __uint32_t MATERIAL_ADVANTAGE_TO_WIN = 500;

// ------------------------------
// Compute Kernels
// ------------------------------

/* Perf test kernels */

__global__ void
SimulateGamesKernel(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves,
                    __int32_t maxDepth);

__global__ void
SimulateGamesKernelSplitMoves(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results,
                              __int32_t maxDepth);

__global__ void PolluteCache(__uint32_t *data, const __uint32_t *seeds, __uint32_t *output, __uint32_t rounds);

/* Engine kernels */

template<__uint32_t SIZE_BOARDS>
__global__ void EvaluateBoardsPlainKernel(
        cuda_PackedBoard<SIZE_BOARDS> *boards,
        const __uint32_t *seeds,
        __uint32_t *results,
        __int32_t maxDepth,
        BYTE *workMem) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    __uint32_t seed = seeds[idx];

    __shared__ __uint32_t counters[SINGLE_THREAD_SINGLE_GAME_BATCH_SIZE];
    int depth{};
    while (depth < maxDepth) {
        const __uint32_t memOffset = idx * DEFAULT_STACK_SIZE * sizeof(cuda_Move);

        /* Prepare stack and move gen */
        Stack<cuda_Move> stack(workMem + memOffset, counters + threadIdx.x);
        MoveGenerator<SIZE_BOARDS> mGen{(*boards)[idx], stack};

        /* Generate moves */
        mGen.GetMovesFast();
        __syncthreads();

        if (stack.Size() == 0) {
            /* We reached end of moves decide about the label and die */
            results[idx] = mGen.EvalBoardsNoMoves();
            return;
        }

        /* Apply random generated move */
        const auto nextMove = stack[seed % stack.Size()];
        cuda_Move::MakeMove(nextMove, (*boards)[idx]);

        /* shuffle the seed */
        simpleRand(seed);
        ++depth;
    }

    results[idx] = DRAW;
}

template<__uint32_t SIZE_BOARDS>
__global__ void EvaluateBoardsSplitKernel(
        cuda_PackedBoard<SIZE_BOARDS> *boards,
        const __uint32_t *seeds,
        __uint32_t *results,
        __int32_t maxDepth,
        BYTE *workMem) {

}


// --------------------------------
// Compute kernels components
// --------------------------------

FAST_CALL_ALWAYS __uint32_t CalcBoardIdx(const __uint32_t tx, const __uint32_t bx, const __uint32_t bs) {
    const __uint32_t plainIdx = bx * bs + tx;
    return (WARP_SIZE * (plainIdx / MINIMAL_SPLIT_BATCH_THREAD_SIZE)) + (tx % WARP_SIZE);
}

FAST_CALL_ALWAYS __uint32_t CalcFigIdx(const __uint32_t tx) {
    return (tx / WARP_SIZE) % BIT_BOARDS_PER_COLOR;
}

FAST_CALL_ALWAYS __uint32_t CalcResourceIdx(const __uint32_t tx) {
    return WARP_SIZE * (tx / MINIMAL_SPLIT_BATCH_THREAD_SIZE) + (tx % WARP_SIZE);
}

FAST_CALL_ALWAYS thrust::tuple<__uint32_t, __uint32_t, __uint32_t, __uint32_t>
CalcSplitIdx(const __uint32_t tx, const __uint32_t bx, const __uint32_t bs) {
    const __uint32_t plainIdx = bx * bs + tx;
    const __uint32_t boardIdx = CalcBoardIdx(tx, bx, bs);
    const __uint32_t figIdx = CalcFigIdx(tx);
    const __uint32_t counterIdx = CalcResourceIdx(tx);

    return {plainIdx, boardIdx, figIdx, counterIdx};
}


#endif //SRC_COMPUTEKERNELS_CUH
