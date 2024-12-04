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
static constexpr __uint32_t SINGLE_THREAD_SINGLE_GAME_STACK_SIZE = 256;
void __global__
SimulateGamesKernel(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves, int maxDepth);

static constexpr __uint32_t SINGLE_THREAD_SINGLE_GAME_SHARED_BATCH_SIZE = 384;

static constexpr __uint32_t WARP_SIZE = 32;
static constexpr __uint32_t MINIMAL_BATCH_SIZE = WARP_SIZE * BIT_BOARDS_PER_COLOR;
static constexpr __uint32_t SINGLE_BATCH_NUM_MINIMAL_BATCHES = 2;
static constexpr __uint32_t SINGLE_BATCH_SIZE = MINIMAL_BATCH_SIZE * SINGLE_BATCH_NUM_MINIMAL_BATCHES;
static constexpr __uint32_t SINGLE_BATCH_BOARD_SIZE = WARP_SIZE * SINGLE_BATCH_NUM_MINIMAL_BATCHES;
static constexpr __uint32_t SINGLE_RUN_BLOCK_SIZE = 100;
static constexpr __uint32_t SINGLE_RUN_BOARDS_SIZE = SINGLE_RUN_BLOCK_SIZE * SINGLE_BATCH_BOARD_SIZE;
static constexpr __uint32_t SPLIT_MAX_STACK_MOVES = 116;

void __global__
SimulateGamesKernelSplitMoves(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, int maxDepth);

__global__ void PolluteCache(__uint32_t *data, const __uint32_t *seeds, __uint32_t *output, __uint32_t rounds);


FAST_CALL_ALWAYS __uint32_t CalcBoardIdx(const __uint32_t tx, const __uint32_t bx, const __uint32_t bs) {
    const __uint32_t plainIdx = bx * bs + tx;
    return (WARP_SIZE * (plainIdx / MINIMAL_BATCH_SIZE)) + (tx % WARP_SIZE);
}

FAST_CALL_ALWAYS __uint32_t CalcFigIdx(const __uint32_t tx) {
    return (tx / WARP_SIZE) % BIT_BOARDS_PER_COLOR;
}

FAST_CALL_ALWAYS __uint32_t CalcResourceIdx(const __uint32_t tx) {
    return WARP_SIZE * (tx / MINIMAL_BATCH_SIZE) + (tx % WARP_SIZE);
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
