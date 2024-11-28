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

void __global__
SimulateGamesKernel(cuda_Board *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves, int maxDepth);

void __global__
SimulateGamesKernelSplitMoves(cuda_Board *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves, int maxDepth);

#endif //SRC_COMPUTEKERNELS_CUH
