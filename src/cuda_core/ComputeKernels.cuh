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

HYBRID thrust::tuple<__uint32_t, __uint32_t, __uint32_t, __uint32_t>
CalcSplitIdx(__uint32_t tx, __uint32_t bx, __uint32_t bs);

static constexpr __uint32_t SINGLE_THREAD_SINGLE_GAME_BATCH_SIZE = 384;
static constexpr __uint32_t SINGLE_THREAD_SINGLE_GAME_STACK_SIZE = 256;
void __global__
SimulateGamesKernel(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves, int maxDepth);

static constexpr __uint32_t SINGLE_THREAD_SINGLE_GAME_SHARED_BATCH_SIZE = 384;


void __global__
SimulateGamesKernelShared(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves,
                          int maxDepth);

static constexpr __uint32_t WARP_SIZE = 32;
static constexpr __uint32_t MINIMAL_BATCH_SIZE = WARP_SIZE * BIT_BOARDS_PER_COLOR;
static constexpr __uint32_t SINGLE_BATCH_NUM_MINIMAL_BATCHES = 2;
static constexpr __uint32_t SINGLE_BATCH_SIZE = MINIMAL_BATCH_SIZE * SINGLE_BATCH_NUM_MINIMAL_BATCHES;
static constexpr __uint32_t SINGLE_BATCH_BOARD_SIZE = WARP_SIZE * SINGLE_BATCH_NUM_MINIMAL_BATCHES;
static constexpr __uint32_t SINGLE_RUN_BLOCK_SIZE = 32 * 2;
static constexpr __uint32_t SINGLE_RUN_BOARDS_SIZE = SINGLE_RUN_BLOCK_SIZE * SINGLE_BATCH_BOARD_SIZE;
static constexpr __uint32_t SPLIT_MAX_STACK_MOVES = 116;

void __global__
SimulateGamesKernelSplitMoves(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, int maxDepth);

void __global__
SimulateGamesKernelSplitMovesShared(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves,
                                    int maxDepth);

__global__ void PolluteCache(__uint32_t *data, const __uint32_t *seeds, __uint32_t *output, __uint32_t rounds);

#endif //SRC_COMPUTEKERNELS_CUH
