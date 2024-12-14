#include "ComputeKernels.cuh"

__global__ void
SimulateGamesKernel(DefaultPackedBoardT *boards, const uint32_t *seeds, uint64_t *results, cuda_Move *moves,
                    int maxDepth) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = seeds[idx];

    __shared__ uint32_t counters[SINGLE_THREAD_SINGLE_GAME_BATCH_SIZE];
    int depth{};
    while (depth < maxDepth) {
        Stack<cuda_Move> stack(moves + idx * DEFAULT_STACK_SIZE, counters + threadIdx.x);

        MoveGenerator mGen{(*boards)[idx], stack};
        mGen.GetMovesFast();
        __syncwarp();

        auto result = (uint32_t *) (results + idx);
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
SimulateGamesKernelSplitMoves(DefaultPackedBoardT *boards, const uint32_t *seeds, uint64_t *results, int maxDepth) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ uint32_t counters[SINGLE_BATCH_BOARD_SIZE];
    __shared__ stub stacks[SINGLE_BATCH_BOARD_SIZE][SPLIT_MAX_STACK_MOVES];

    uint32_t seed = seeds[CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x)];
    while (maxDepth-- > 0) {
        __syncthreads();

        Stack<cuda_Move> stack((cuda_Move *) stacks[CalcResourceIdx(threadIdx.x)],
                               counters + CalcResourceIdx(threadIdx.x), false);

        if (CalcFigIdx(threadIdx.x) == 0) {
            stack.Clear();
        }

        __syncthreads();

        MoveGenerator<PACKED_BOARD_DEFAULT_SIZE, SPLIT_MAX_STACK_MOVES> mGen{
            (*boards)[CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x)], stack
        };
        mGen.GetMovesSplit(CalcFigIdx(threadIdx.x));
        __syncthreads();

        if (CalcFigIdx(threadIdx.x) == 0) {
            auto result = (uint32_t *) (results + CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x));
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

__global__ void PolluteCache(uint32_t *data, const uint32_t *seeds, uint32_t *output, uint32_t rounds) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = seeds[idx];

    for (uint32_t round = 0; round < rounds; ++round) {
        output[idx] += data[idx];
        data[idx] += data[idx];
        data[idx] -= output[idx];

        simpleRand(seed);
    }
}

__global__ void EvaluateBoardsSplitKernel(cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS> *boards, const uint32_t *seeds,
                                          uint32_t *results, int32_t maxDepth, BYTE *workMem) {
    struct stub {
        char bytes[sizeof(cuda_Move)];
    };

    __shared__ uint32_t counters[EVAL_SPLIT_KERNEL_BOARDS];
    __shared__ stub stacks[EVAL_SPLIT_KERNEL_BOARDS][SPLIT_MAX_STACK_MOVES];

    const uint32_t boardIdx = CalcBoardIdx(threadIdx.x, blockIdx.x, blockDim.x);
    const uint32_t figIdx = CalcFigIdx(threadIdx.x);
    const uint32_t resourceIdx = CalcResourceIdx(threadIdx.x);

    uint32_t seed = seeds[boardIdx];

    while (maxDepth-- > 0) {
        Stack<cuda_Move> stack(reinterpret_cast<cuda_Move *>(stacks + resourceIdx), counters + resourceIdx, false);

        if (figIdx == 0) {
            stack.Clear();
        }

        __syncthreads();

        MoveGenerator<EVAL_SPLIT_KERNEL_BOARDS, SPLIT_MAX_STACK_MOVES> mGen{(*boards)[boardIdx], stack};
        mGen.GetMovesSplit(figIdx);
        __syncthreads();

        const uint32_t movingColor = (*boards)[boardIdx].MovingColor();
        if (stack.Size() == 0) {
            if (figIdx == 0) {
                /* We reached end of moves decide about the label and die */
                results[boardIdx] = mGen.EvalBoardsNoMoves(movingColor);
            }
            return;
        }

        if (figIdx == 0) {
            const auto nextMove = stack[seed % stack.Size()];
            cuda_Move::MakeMove<EVAL_SPLIT_KERNEL_BOARDS>(nextMove, (*boards)[boardIdx]);
        }

        if ((*boards)[boardIdx].HalfMoves() >= HALF_MOVES_TO_DRAW) {
            if (figIdx == 0) {
                results[boardIdx] = DRAW;
            }
            return;
        }

        simpleRand(seed);
    }

    if (figIdx == 0) {
        static constexpr int32_t MIN_EVAL_TO_WIN =
                FIG_VALUES[W_ROOK_INDEX] + FIG_VALUES[W_QUEEN_INDEX] + FIG_VALUES[W_PAWN_INDEX];
        const int32_t eval = (*boards)[boardIdx].EvaluateMaterial();

        if (abs(eval) >= MIN_EVAL_TO_WIN) {
            results[boardIdx] = eval > 0 ? WHITE_WIN : BLACK_WIN;
        } else {
            results[boardIdx] = DRAW;
        }
    }
}
