//
// Created by Jlisowskyy on 28/11/24.
//

#include "ComputeKernels.cuh"

__global__ void __global__
SimulateGamesKernel(cuda_Board *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves, int maxDepth) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    __uint32_t seed = seeds[idx];

    int depth{};

    while (depth < maxDepth) {
        Stack<cuda_Move> stack(moves + idx * 256);
        MoveGenerator mGen{*(boards + idx), stack};
        const auto generatedMoves = mGen.GetMovesFast();
        __syncthreads();

        auto result = (__uint32_t *) (results + idx);
        ++result[0];
        result[1] += generatedMoves.size;

        if (generatedMoves.size == 0) {
            ++depth;
            continue;
        }

        const auto nextMove = generatedMoves[seed % generatedMoves.size];
        cuda_Move::MakeMove(nextMove, *(boards + idx));

        simpleRand(seed);
        ++depth;
    }
}

static constexpr __uint32_t WARP_SIZE = 32;
static constexpr __uint32_t THREADS_PER_PACKAGE = WARP_SIZE * BIT_BOARDS_PER_COLOR;

__global__ void
SimulateGamesKernelSplitMoves(cuda_Board *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves,
                              int maxDepth) {
    const __uint32_t plainIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const __uint32_t boardIdx = (WARP_SIZE * (plainIdx) / THREADS_PER_PACKAGE) + (threadIdx.x % WARP_SIZE);
    const __uint32_t figIdx = (threadIdx.x / WARP_SIZE) % BIT_BOARDS_PER_COLOR;

    __uint32_t seed = seeds[boardIdx];

    int depth{};

    while (depth < maxDepth) {
        __syncthreads();

        Stack<cuda_Move> stack(moves + boardIdx * 256);

        if (figIdx == 0) {
            stack.Clear();
        }

        __syncthreads();

        MoveGenerator mGen{*(boards + boardIdx), stack};
        mGen.GetMovesSplit(figIdx);
        __syncthreads();

        if (figIdx == 0) {
            auto result = (__uint32_t *) (results + boardIdx);
            ++result[0];
            result[1] += stack.Size();
        }

        if (stack.Size() == 0) {
            ++depth;
            continue;
        }

        if (figIdx == 0) {
            const auto nextMove = stack[seed % stack.Size()];
            cuda_Move::MakeMove(nextMove, *(boards + boardIdx));
        }

        simpleRand(seed);
        ++depth;
    }
}
