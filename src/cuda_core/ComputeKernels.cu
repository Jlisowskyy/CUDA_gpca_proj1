//
// Created by Jlisowskyy on 28/11/24.
//

#include "ComputeKernels.cuh"

__global__ void
SimulateGamesKernel(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves, int maxDepth) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    __uint32_t seed = seeds[idx];

    __shared__ __uint32_t counters[SINGLE_THREAD_SINGLE_GAME_BATCH_SIZE];
    int depth{};
    while (depth < maxDepth) {
        Stack<cuda_Move> stack(moves + idx * DEFAULT_STACK_SIZE, counters + threadIdx.x);

        MoveGenerator mGen{(*boards)[idx], stack};
        mGen.GetMovesFast();
        __syncthreads();

        auto result = (__uint32_t *) (results + idx);
        ++result[0];
        result[1] += stack.Size();

        if (stack.Size() == 0) {
            ++depth;
            continue;
        }

        const auto nextMove = stack[seed % stack.Size()];
        cuda_Move::MakeMove(nextMove, (*boards)[idx]);

        simpleRand(seed);
        ++depth;
    }
}

__global__ void
SimulateGamesKernelSplitMoves(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, int maxDepth) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ __uint32_t counters[SINGLE_BATCH_BOARD_SIZE];
    __shared__ stub stacks[SINGLE_BATCH_BOARD_SIZE][SPLIT_MAX_STACK_MOVES];

    __uint32_t seed = seeds[CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x)];
    while (maxDepth --> 0) {
        __syncthreads();

        Stack<cuda_Move> stack((cuda_Move*)stacks[CalcResourceIdx(threadIdx.x)], counters + CalcResourceIdx(threadIdx.x), false);

        if (CalcFigIdx(threadIdx.x) == 0) {
            stack.Clear();
        }

        __syncthreads();

        MoveGenerator<PACKED_BOARD_DEFAULT_SIZE, SPLIT_MAX_STACK_MOVES> mGen{(*boards)[CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x)], stack};
        mGen.GetMovesSplit(CalcFigIdx(threadIdx.x));
        __syncthreads();

        if (CalcFigIdx(threadIdx.x) == 0) {
            auto result = (__uint32_t *) (results + CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x));
            ++result[0];
            result[1] += stack.Size();
        }

        if (stack.Size() == 0) {
            continue;
        }

        if (CalcFigIdx(threadIdx.x) == 0) {
            const auto nextMove = stack[seed % stack.Size()];
            cuda_Move::MakeMove(nextMove, (*boards)[CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x)]);
        }

        simpleRand(seed);
    }
}

__global__ void PolluteCache(__uint32_t *data, const __uint32_t *seeds, __uint32_t *output, __uint32_t rounds) {
    const __uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __uint32_t seed = seeds[idx];

    for (__uint32_t round = 0; round < rounds; ++round) {
        output[idx] += data[idx];
        data[idx] += data[idx];
        data[idx] -= output[idx];

        simpleRand(seed);
    }
}

__global__ void EvaluateBoardsSplitKernel(cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS> *boards, const __uint32_t *seeds,
                                          __uint32_t *results, __int32_t maxDepth, BYTE *workMem) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ __uint32_t counters[EVAL_SPLIT_KERNEL_BOARDS];
    __shared__ stub stacks[EVAL_SPLIT_KERNEL_BOARDS][SPLIT_MAX_STACK_MOVES];
    __shared__ __uint16_t evalCounters[EVAL_SPLIT_KERNEL_BOARDS][2];

    const __uint32_t boardIdx = CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x);
    const __uint32_t figIdx = CalcFigIdx(threadIdx.x);
    const __uint32_t resourceIdx = CalcResourceIdx(threadIdx.x);

    if (figIdx== 0) {
        evalCounters[resourceIdx][1] = 0;
        evalCounters[resourceIdx][0] = 0;
    }

    __uint32_t seed = seeds[boardIdx];

    while (maxDepth --> 0) {
        __syncthreads();

        Stack<cuda_Move> stack((cuda_Move*)stacks[resourceIdx], counters + resourceIdx, false);

        if (figIdx == 0) {
            stack.Clear();
        }

        __syncthreads();

        MoveGenerator<EVAL_SPLIT_KERNEL_BOARDS, SPLIT_MAX_STACK_MOVES> mGen{(*boards)[boardIdx], stack};
        mGen.GetMovesSplit(figIdx);
        __syncthreads();

        const __uint32_t movingColor = boards->operator[](boardIdx).MovingColor();
        if (stack.Size() == 0) {
            if (figIdx == 0) {
                /* We reached end of moves decide about the label and die */
                results[boardIdx] = mGen.EvalBoardsNoMoves(movingColor);
                return;
            } else {
                return;
            }
        }

        if (figIdx == 0) {
            /* Check if board is enough rounds in winning position to assume that's a win */
            __int32_t eval = (*boards)[boardIdx].MaterialEval();
            eval = movingColor == BLACK ? -eval : eval;
            const bool isInWinningRange = eval >= MATERIAL_ADVANTAGE_TO_WIN;

            evalCounters[resourceIdx][movingColor] = isInWinningRange ? evalCounters[resourceIdx][movingColor] + 1: 0;
        }

        __syncthreads();

        /* ASSUME win and die */
        if (evalCounters[resourceIdx][movingColor] >= NUM_ROUNDS_IN_MATERIAL_ADVANTAGE_TO_WIN) {
            if (figIdx == 0) {
                results[boardIdx] = movingColor;
            }

            return;
        }

        if (figIdx == 0) {
            const auto nextMove = stack[seed % stack.Size()];
            cuda_Move::MakeMove<EVAL_SPLIT_KERNEL_BOARDS>(nextMove, (*boards)[boardIdx]);
        }

        simpleRand(seed);
    }
}

