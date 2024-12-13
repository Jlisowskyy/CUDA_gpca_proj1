#ifndef SRC_COMPUTEKERNELS_CUH
#define SRC_COMPUTEKERNELS_CUH

#include "Helpers.cuh"
#include "cuda_Board.cuh"
#include "Move.cuh"
#include "MoveGenerator.cuh"
#include "cuda_Board.cuh"
#include "cuda_PackedBoard.cuh"

#include <thrust/tuple.h>


static constexpr uint32_t SINGLE_THREAD_SINGLE_GAME_BATCH_SIZE = 384;
static constexpr uint32_t SINGLE_THREAD_SINGLE_GAME_STACK_SIZE = DEFAULT_STACK_SIZE;

static constexpr uint32_t WARP_SIZE = 32;
static constexpr uint32_t MINIMAL_SPLIT_BATCH_THREAD_SIZE = WARP_SIZE * BIT_BOARDS_PER_COLOR;
static constexpr uint32_t SPLIT_BATCH_NUM_MINIMALS = 2;
static constexpr uint32_t SPLIT_BATCH_NUM_THREADS = MINIMAL_SPLIT_BATCH_THREAD_SIZE * SPLIT_BATCH_NUM_MINIMALS;
static constexpr uint32_t SINGLE_BATCH_BOARD_SIZE = WARP_SIZE * SPLIT_BATCH_NUM_MINIMALS;
static constexpr uint32_t SINGLE_RUN_BLOCK_SIZE = 100;

static constexpr uint32_t SINGLE_RUN_BOARDS_SIZE = SINGLE_RUN_BLOCK_SIZE * SINGLE_BATCH_BOARD_SIZE;
static constexpr uint32_t SPLIT_MAX_STACK_MOVES = 80;

static __device__ __constant__ constexpr uint32_t NUM_ROUNDS_IN_MATERIAL_ADVANTAGE_TO_WIN = 30;
static __device__ __constant__ constexpr uint32_t MATERIAL_ADVANTAGE_TO_WIN = 700;
static __device__ __constant__ constexpr uint32_t HALF_MOVES_TO_DRAW = 50;

// --------------------------------
// Compute kernels components
// --------------------------------

FAST_CALL_ALWAYS uint32_t CalcBoardIdx(const uint32_t tx, const uint32_t bx, const uint32_t bs) {
    const uint32_t plainIdx = bx * bs + tx;
    return (WARP_SIZE * (plainIdx / MINIMAL_SPLIT_BATCH_THREAD_SIZE)) + (tx % WARP_SIZE);
}

FAST_CALL_ALWAYS uint32_t CalcFigIdx(const uint32_t tx) {
    return (tx / WARP_SIZE) % BIT_BOARDS_PER_COLOR;
}

FAST_CALL_ALWAYS uint32_t CalcResourceIdx(const uint32_t tx) {
    return WARP_SIZE * (tx / MINIMAL_SPLIT_BATCH_THREAD_SIZE) + (tx % WARP_SIZE);
}

FAST_CALL_ALWAYS thrust::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
CalcSplitIdx(const uint32_t tx, const uint32_t bx, const uint32_t bs) {
    const uint32_t plainIdx = bx * bs + tx;
    const uint32_t boardIdx = CalcBoardIdx(tx, bx, bs);
    const uint32_t figIdx = CalcFigIdx(tx);
    const uint32_t counterIdx = CalcResourceIdx(tx);

    return {plainIdx, boardIdx, figIdx, counterIdx};
}


// ------------------------------
// Compute Kernels
// ------------------------------

/* Perf test kernels */

__global__ void
SimulateGamesKernel(DefaultPackedBoardT *boards, const uint32_t *seeds, uint64_t *results, cuda_Move *moves,
                    int32_t maxDepth);

__global__ void
SimulateGamesKernelSplitMoves(DefaultPackedBoardT *boards, const uint32_t *seeds, uint64_t *results,
                              int32_t maxDepth);

__global__ void PolluteCache(uint32_t *data, const uint32_t *seeds, uint32_t *output, uint32_t rounds);

/* Engine kernels */

#ifdef NDEBUG

static constexpr uint32_t EVAL_PLAIN_KERNEL_BOARDS = 384;

#else

static constexpr uint32_t EVAL_PLAIN_KERNEL_BOARDS = 384 / 2;

#endif

template<uint32_t SIZE_BOARDS>
__global__ void EvaluateBoardsPlainKernel(
    cuda_PackedBoard<SIZE_BOARDS> *boards,
    const uint32_t *seeds,
    uint32_t *results,
    int32_t maxDepth,
    BYTE *workMem) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = seeds[idx];

    __shared__ uint32_t counters[SIZE_BOARDS];

    int depth{};
    while (depth < maxDepth) {
        const uint32_t memOffset = idx * DEFAULT_STACK_SIZE * sizeof(cuda_Move);

        /* Prepare stack and move gen */
        Stack<cuda_Move> stack(workMem + memOffset, counters + threadIdx.x);
        MoveGenerator<SIZE_BOARDS> mGen{(*boards)[idx], stack};

        /* Generate moves */
        mGen.GetMovesFast();
        __syncthreads();

        const uint32_t movingColor = (*boards)[idx].MovingColor();
        if (stack.Size() == 0) {
            /* We reached end of moves decide about the label and die */
            results[idx] = mGen.EvalBoardsNoMoves(movingColor);
            return;
        }

        if ((*boards)[idx].HalfMoves() >= HALF_MOVES_TO_DRAW) {
            results[idx] = DRAW;
            return;
        }

        /* Apply random generated move */
        const auto nextMove = stack[seed % stack.Size()];
        cuda_Move::MakeMove<SIZE_BOARDS>(nextMove, (*boards)[idx]);

        /* shuffle the seed */
        simpleRand(seed);
        ++depth;
    }

    results[idx] = DRAW;
}

#ifdef NDEBUG

static constexpr uint32_t EVAL_SPLIT_KERNEL_BOARDS = 640;

#else

static constexpr uint32_t EVAL_SPLIT_KERNEL_BOARDS = 32;

#endif

__global__ void EvaluateBoardsSplitKernel(
    cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS> *boards,
    const uint32_t *seeds,
    uint32_t *results,
    int32_t maxDepth,
    [[maybe_unused]] BYTE *workMem);


#endif //SRC_COMPUTEKERNELS_CUH
