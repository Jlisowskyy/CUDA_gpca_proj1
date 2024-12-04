//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_MCTSENGINE_CUH
#define SRC_MCTSENGINE_CUH

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"

#include "MctsTree.cuh"
#include "cpu_MoveGen.cuh"

#include <random>
#include <thread>

static constexpr __uint32_t DEFAULT_MCTS_BATCH_SIZE = 64;

template<__uint32_t BATCH_SIZE>
class MctsEngine {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    explicit MctsEngine(const cuda_Board &board) : m_board(board) {}

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] cuda_Move MoveSearch(const __uint32_t moveTime) {
        auto moves = ported_translation::GenMoves(m_board);
        std::shuffle(moves.begin(), moves.end(), std::mt19937_64(std::random_device()()));

        std::this_thread::sleep_for(std::chrono::milliseconds(moveTime));

        return moves[0];
    }

    void ApplyMove(const cuda_Move move) {
        assert(move.IsOkayMoveCPU() && "ENGINE RECEIVED MALFUNCTIONING MOVE!");


        cuda_Move::MakeMove(move, m_board);

        /* adapt subtree */
    }

    // ------------------------------
    // Class implementation
    // ------------------------------
protected:

    // ------------------------------
    // Class fields
    // ------------------------------

    __uint32_t m_moveTime{};
    cuda_Board m_board{};
};


#endif //SRC_MCTSENGINE_CUH
