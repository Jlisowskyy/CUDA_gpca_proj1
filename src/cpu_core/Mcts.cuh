//
// Created by Jlisowskyy on 05/12/24.
//

#ifndef SRC_MCTS_CUH
#define SRC_MCTS_CUH

#include "MctsNode.cuh"

namespace mcts {
    template<__uint32_t BATCH_SIZE>
    void ExpandTreeGPU(MctsNode *root) {

    }

    void ExpandTreeCPU(MctsNode *root);
}

#endif //SRC_MCTS_CUH
