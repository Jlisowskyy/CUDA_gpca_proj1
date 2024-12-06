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

namespace mcts {
    template<__uint32_t BATCH_SIZE>
    using results_t = std::array<__uint32_t, BATCH_SIZE>;

    template<__uint32_t BATCH_SIZE>
    using sim_t = results_t<BATCH_SIZE> (*)(const cuda_PackedBoard<BATCH_SIZE> &, cudaStream_t &);

    extern std::atomic<__uint32_t> g_ExpandRacesCounter;
    extern std::atomic<__uint64_t> g_SimulationCounter;

    void ExpandTreeCPU(MctsNode *root);

    [[nodiscard]] MctsNode *SelectNode(MctsNode *root);

    [[nodiscard]] MctsNode *ExpandNode(MctsNode *root);

    void PropagateResult(MctsNode *node, __uint32_t result);

    inline results_t<EVAL_SPLIT_KERNEL_BOARDS>
    SimulateSplit(const cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS> &boards, cudaStream_t &stream) {
        results_t<EVAL_SPLIT_KERNEL_BOARDS> hResults{};

        __uint32_t *dSeeds{};
        __uint32_t *dResults{};
        cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS> *dBoards{};

        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dBoards, sizeof(cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS>), stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dSeeds, sizeof(__uint32_t) * EVAL_SPLIT_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dResults, sizeof(__uint32_t) * EVAL_SPLIT_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dBoards, &boards, sizeof(cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS>),
                                            cudaMemcpyHostToDevice, stream));

        const auto seeds = GenSeeds(EVAL_SPLIT_KERNEL_BOARDS);
        assert(seeds.size() == EVAL_SPLIT_KERNEL_BOARDS);

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dSeeds, seeds.data(), sizeof(__uint32_t) * EVAL_SPLIT_KERNEL_BOARDS,
                                            cudaMemcpyHostToDevice, stream));

        EvaluateBoardsSplitKernel<<<EVAL_SPLIT_KERNEL_BOARDS / WARP_SIZE, WARP_SIZE *
                                                                          BIT_BOARDS_PER_COLOR, 0, stream>>>(
                dBoards, dSeeds, dResults, MAX_SIMULATION_DEPTH, nullptr
        );

//        EvaluateBoardsSplitKernel<<<1, EVAL_SPLIT_KERNEL_BOARDS * BIT_BOARDS_PER_COLOR, 0, stream>>>(
//                dBoards, dSeeds, dResults, MAX_SIMULATION_DEPTH, nullptr
//        );

        CUDA_ASSERT_SUCCESS(cudaGetLastError());

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(hResults.data(), dResults,
                                            sizeof(__uint32_t) * EVAL_SPLIT_KERNEL_BOARDS, cudaMemcpyDeviceToHost,
                                            stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dSeeds, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBoards, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dResults, stream));

        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));

        return hResults;
    }

    template<__uint32_t BATCH_SIZE>
    results_t<BATCH_SIZE> SimulatePlain(const cuda_PackedBoard<BATCH_SIZE> &boards, cudaStream_t &stream) {
        throw std::runtime_error("NOT IMPLEMENTED");
    }

    template<__uint32_t BATCH_SIZE = EVAL_SPLIT_KERNEL_BOARDS, sim_t<BATCH_SIZE> FUNC = SimulateSplit>
    void ExpandTreeGPU(MctsNode *root, cudaStream_t &stream) {
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
                    continue;
                }

                /* we have proper node selected continue work ... */
                node = expandedNode;
            }

            selectedNodes[idx] = node;
            batch.saveBoard(idx, node->m_board);
        }

        const auto results = FUNC(batch, stream);

        assert(results.size() == selectedNodes.size());
        for (__uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            PropagateResult(selectedNodes[idx], results[idx]);
        }

        g_SimulationCounter.fetch_add(EVAL_SPLIT_KERNEL_BOARDS, std::memory_order::relaxed);
    }
}

#endif //SRC_MCTS_CUH
