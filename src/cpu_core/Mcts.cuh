//
// Created by Jlisowskyy on 05/12/24.
//

#ifndef SRC_MCTS_CUH
#define SRC_MCTS_CUH

#include "MctsNode.cuh"
#include "../cuda_core/cuda_PackedBoard.cuh"

#include <array>
#include <atomic>

static constexpr __uint32_t MIN_SAMPLES_TO_EXPAND = 32;

namespace mcts {

    extern std::atomic<__uint32_t> ExpandRacesCounter;
    extern std::atomic<__uint64_t> SimulationCounter;

    void ExpandTreeCPU(MctsNode *root);

    [[nodiscard]] MctsNode *SelectNode(MctsNode *root);

    [[nodiscard]] MctsNode *ExpandNode(MctsNode *root);

    void PropagateResult(MctsNode *node, __uint32_t result);

    template<__uint32_t BATCH_SIZE>
    std::array<__uint32_t, BATCH_SIZE> Simulate(const cuda_PackedBoard<BATCH_SIZE> &boards) {

    }

    template<__uint32_t BATCH_SIZE>
    void ExpandTreeGPU(MctsNode *root) {
        cuda_PackedBoard<BATCH_SIZE> batch;
        std::array<MctsNode *, BATCH_SIZE> selectedNodes;

        for (__uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            auto node = SelectNode(root);
            node = ExpandNode(node);

            selectedNodes[idx] = node;
            batch.saveBoard(idx, node->m_board);
        }

        const auto results = Simulate(batch);

        assert(results.size() == selectedNodes.size());
        for (__uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            PropagateResult(selectedNodes[idx], results[idx]);
        }
    }


}

#endif //SRC_MCTS_CUH
