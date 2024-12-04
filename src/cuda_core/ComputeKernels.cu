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

