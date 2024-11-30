//
// Created by Jlisowskyy on 28/11/24.
//

#include "ComputeKernels.cuh"

__global__ void
SimulateGamesKernel(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves, int maxDepth) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    __uint32_t seed = seeds[idx];

    int depth{};

    while (depth < maxDepth) {
        Stack<cuda_Move> stack(moves + idx * 256);
        MoveGenerator mGen{(*boards)[idx], stack};
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
        cuda_Move::MakeMove(nextMove, (*boards)[idx]);

        simpleRand(seed);
        ++depth;
    }
}

__global__ void
SimulateGamesKernelShared(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves,
                          int maxDepth) {
//    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
//    __uint32_t seed = seeds[idx];
//
//    int depth{};
//
//    while (depth < maxDepth) {
//        Stack<cuda_Move> stack(moves + idx * 256);
//        MoveGenerator mGen{*(boards + idx), stack};
//        const auto generatedMoves = mGen.GetMovesFast();
//        __syncthreads();
//
//        auto result = (__uint32_t *) (results + idx);
//        ++result[0];
//        result[1] += generatedMoves.size;
//
//        if (generatedMoves.size == 0) {
//            ++depth;
//            continue;
//        }
//
//        const auto nextMove = generatedMoves[seed % generatedMoves.size];
//        cuda_Move::MakeMove(nextMove, *(boards + idx));
//
//        simpleRand(seed);
//        ++depth;
//    }
}

__global__ void
SimulateGamesKernelSplitMoves(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves,
                              int maxDepth) {
    const __uint32_t plainIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const __uint32_t boardIdx = (WARP_SIZE * (plainIdx / MINIMAL_BATCH_SIZE)) + (threadIdx.x % WARP_SIZE);
    const __uint32_t figIdx = (threadIdx.x / WARP_SIZE) % BIT_BOARDS_PER_COLOR;

    __uint32_t seed = seeds[boardIdx];

    int depth{};

    while (depth < maxDepth) {
        __syncthreads();

        Stack<cuda_Move> stack(moves + boardIdx * 256 + 2 /* 2 * sizeof(move) == 16 */, false);
        auto *md = (MoveGenDataMem *) (moves + boardIdx * 256);

        if (figIdx == 0) {
            stack.Clear();

            md->checksCount = 0;
            md->blockedMap = 0;
            md->wasCheckedBySimple = 0;
        }

        __syncthreads();

        MoveGenerator mGen{(*boards)[boardIdx], stack, md};
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
            cuda_Move::MakeMove(nextMove, (*boards)[boardIdx]);
        }

        simpleRand(seed);
        ++depth;
    }
}

__global__ void
SimulateGamesKernelSplitMovesShared(DefaultPackedBoardT *boards, const __uint32_t *seeds, __uint64_t *results, cuda_Move *moves,
                                    int maxDepth) {
//    const __uint32_t plainIdx = blockIdx.x * blockDim.x + threadIdx.x;
//    const __uint32_t boardIdx = (WARP_SIZE * (plainIdx / MINIMAL_BATCH_SIZE)) + (threadIdx.x % WARP_SIZE);
//    const __uint32_t figIdx = (threadIdx.x / WARP_SIZE) % BIT_BOARDS_PER_COLOR;
//
//    __shared__ cuda_Board sharedBoards[SINGLE_BATCH_BOARD_SIZE];
//
//    switch (figIdx) {
//        case PAWN_INDEX:
//            sharedBoards[threadIdx.x].BitBoards[W_PAWN_INDEX] = boards[boardIdx].BitBoards[W_PAWN_INDEX];
//            sharedBoards[threadIdx.x].BitBoards[B_PAWN_INDEX] = boards[boardIdx].BitBoards[B_PAWN_INDEX];
//            sharedBoards[threadIdx.x].MovingColor = boards[boardIdx].MovingColor;
//            break;
//        case KNIGHT_INDEX:
//            sharedBoards[threadIdx.x].BitBoards[W_KNIGHT_INDEX] = boards[boardIdx].BitBoards[W_KNIGHT_INDEX];
//            sharedBoards[threadIdx.x].BitBoards[B_KNIGHT_INDEX] = boards[boardIdx].BitBoards[B_KNIGHT_INDEX];
//            sharedBoards[threadIdx.x].Castlings = boards[boardIdx].Castlings;
//            break;
//        case BISHOP_INDEX:
//            sharedBoards[threadIdx.x].BitBoards[W_BISHOP_INDEX] = boards[boardIdx].BitBoards[W_BISHOP_INDEX];
//            sharedBoards[threadIdx.x].BitBoards[B_BISHOP_INDEX] = boards[boardIdx].BitBoards[B_BISHOP_INDEX];
//            sharedBoards[threadIdx.x].ElPassantField = boards[boardIdx].ElPassantField;
//            break;
//        case ROOK_INDEX:
//            sharedBoards[threadIdx.x].BitBoards[W_ROOK_INDEX] = boards[boardIdx].BitBoards[W_ROOK_INDEX];
//            sharedBoards[threadIdx.x].BitBoards[B_ROOK_INDEX] = boards[boardIdx].BitBoards[B_ROOK_INDEX];
//            break;
//        case QUEEN_INDEX:
//            sharedBoards[threadIdx.x].BitBoards[W_QUEEN_INDEX] = boards[boardIdx].BitBoards[W_QUEEN_INDEX];
//            sharedBoards[threadIdx.x].BitBoards[B_QUEEN_INDEX] = boards[boardIdx].BitBoards[B_QUEEN_INDEX];
//            break;
//        case KING_INDEX:
//            sharedBoards[threadIdx.x].BitBoards[W_KING_INDEX] = boards[boardIdx].BitBoards[W_KING_INDEX];
//            sharedBoards[threadIdx.x].BitBoards[B_KING_INDEX] = boards[boardIdx].BitBoards[B_KING_INDEX];
//            break;
//        default:
//            ASSERT(false, "Shit happens");
//    }
//
//    __uint32_t seed = seeds[boardIdx];
//    int depth{};
//
//    __syncthreads();
//    while (depth < maxDepth) {
//        __syncthreads();
//
//        Stack<cuda_Move> stack(moves + boardIdx * 256 + 2 /* 2 * sizeof(move) == 16 */, false);
//        auto *md = (MoveGenDataMem *) (moves + boardIdx * 256);
//
//        if (figIdx == 0) {
//            stack.Clear();
//
//            md->checksCount = 0;
//            md->blockedMap = 0;
//            md->wasCheckedBySimple = 0;
//        }
//
//        __syncthreads();
//
//        MoveGenerator mGen{*(sharedBoards + threadIdx.x), stack, md};
//        mGen.GetMovesSplit(figIdx);
//        __syncthreads();
//
//        if (figIdx == 0) {
//            auto result = (__uint32_t *) (results + boardIdx);
//            ++result[0];
//            result[1] += stack.Size();
//        }
//
//        if (stack.Size() == 0) {
//            ++depth;
//            continue;
//        }
//
//        if (figIdx == 0) {
//            const auto nextMove = stack[seed % stack.Size()];
//            cuda_Move::MakeMove(nextMove, *(sharedBoards + threadIdx.x));
//        }
//
//        simpleRand(seed);
//        ++depth;
//    }
}
