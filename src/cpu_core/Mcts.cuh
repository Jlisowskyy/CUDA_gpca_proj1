//
// Created by Jlisowskyy on 05/12/24.
//

#ifndef SRC_MCTS_CUH
#define SRC_MCTS_CUH

#include "MctsNode.cuh"
#include "../cuda_core/cuda_PackedBoard.cuh"

#include <array>

static constexpr __uint32_t MIN_SAMPLES_TO_EXPAND = 32;

namespace mcts {

    void ExpandTreeCPU(MctsNode *root);

    [[nodiscard]] MctsNode *SelectNode(MctsNode *root);

    template<__uint32_t BATCH_SIZE>
    void ExpandTreeGPU(MctsNode *root) {
        cuda_PackedBoard<BATCH_SIZE> batch;
        std::array<MctsNode *, BATCH_SIZE> selectedNodes;

        for (__uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            const auto node = SelectNode(root);

            selectedNodes[idx] = node;
            batch.saveBoard(idx, node->m_board);
        }
    }


}

#endif //SRC_MCTS_CUH
