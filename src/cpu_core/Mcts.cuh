//
// Created by Jlisowskyy on 05/12/24.
//

#ifndef SRC_MCTS_CUH
#define SRC_MCTS_CUH

#include "MctsNode.cuh"
#include "../cuda_core/cuda_PackedBoard.cuh"
#include "../tests/CudaTests.cuh"

#include "../cuda_core/ComputeKernels.cuh"
#include "cpu_MoveGen.cuh"

#include <cuda_runtime.h>

#include <array>
#include <atomic>

static constexpr __uint32_t MIN_SAMPLES_TO_EXPAND = 32;
static constexpr __uint32_t MAX_SIMULATION_DEPTH = 100;

enum class EngineType {
    CPU,
    GPU0,
    GPU1,
};

namespace mcts {
    template<__uint32_t BATCH_SIZE>
    using results_t = std::array<__uint32_t, BATCH_SIZE>;

    template<__uint32_t BATCH_SIZE>
    using sim_t = results_t<BATCH_SIZE> (*)(const cuda_PackedBoard<BATCH_SIZE> &, cudaStream_t &);

    extern std::atomic<__uint32_t> g_ExpandRacesCounter;
    extern std::atomic<__uint64_t> g_SimulationCounter;

    void ExpandTreeCPU(__uint32_t idx, MctsNode *root);

    [[nodiscard]] MctsNode *SelectNode(MctsNode *root);

    [[nodiscard]] MctsNode *ExpandNode(MctsNode *root);

    void PropagateResult(MctsNode *node, __uint32_t result);

    results_t<EVAL_SPLIT_KERNEL_BOARDS>
    SimulateSplit(const cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS> &boards, cudaStream_t &stream);

    results_t<EVAL_PLAIN_KERNEL_BOARDS>
    SimulatePlain(const cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS> &boards, cudaStream_t &stream);

    template<EngineType ENGINE_TYPE>
    void ExpandTreeGPU(MctsNode *root, cudaStream_t &stream) {
        static_assert(ENGINE_TYPE == EngineType::GPU1 || ENGINE_TYPE == EngineType::GPU0);
        static constexpr __uint32_t BATCH_SIZE = ENGINE_TYPE == EngineType::GPU0 ? EVAL_SPLIT_KERNEL_BOARDS :
                                                 EVAL_PLAIN_KERNEL_BOARDS;

        cuda_PackedBoard<BATCH_SIZE> batch;
        std::array<MctsNode *, BATCH_SIZE> selectedNodes;

        for (__uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            MctsNode *node{};

            while (!node) {
                node = SelectNode(root);

                /* We selected node already extended but without any children roll back */
                if (node->HasChildrenAssigned()) {
                    PropagateResult(node,
                                    ported_translation::IsCheck(node->m_board) ?
                                    SwapColor(node->m_board.MovingColor) : DRAW
                    );
                    node = nullptr;
                    g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
                    continue;
                }

                MctsNode *expandedNode = ExpandNode(node);

                if (!expandedNode) {
                    /* Selected node was not expanded yet, but we found out that it is a dead end indeed */
                    PropagateResult(node,
                                    ported_translation::IsCheck(node->m_board) ?
                                    SwapColor(node->m_board.MovingColor) : DRAW
                    );
                    node = nullptr;
                    g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
                    continue;
                }

                /* we have proper node selected continue work ... */
                node = expandedNode;
            }

            selectedNodes[idx] = node;
            batch.saveBoard(idx, node->m_board);
        }

        results_t<BATCH_SIZE> results;

        if constexpr (ENGINE_TYPE == EngineType::GPU0) {
            results = SimulateSplit(batch, stream);
            g_SimulationCounter.fetch_add(EVAL_SPLIT_KERNEL_BOARDS, std::memory_order::relaxed);
        } else {
            results = SimulatePlain(batch, stream);
            g_SimulationCounter.fetch_add(EVAL_PLAIN_KERNEL_BOARDS, std::memory_order::relaxed);
        }

        assert(results.size() == selectedNodes.size());
        for (__uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            PropagateResult(selectedNodes[idx], results[idx]);
        }
    }
}

#endif //SRC_MCTS_CUH
