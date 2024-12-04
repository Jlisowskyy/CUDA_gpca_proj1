//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_MCTSENGINE_CUH
#define SRC_MCTSENGINE_CUH

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"

#include "MctsTree.cuh"

template<__uint32_t BATCH_SIZE>
class MctsEngine {
    // ------------------------------
    // Class creation
    // ------------------------------

    explicit MctsEngine(const cuda_Board &board) : m_board(board) {}

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] cuda_Move MoveSearch(const __uint32_t moveTime) {

    }

    void ApplyMove(const cuda_Move move) {

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
