//
// Created by Jlisowskyy on 05/12/24.
//

#include "Mcts.cuh"

#include "cpu_MoveGen.cuh"

#include <numeric>

namespace mcts {
    std::atomic<__uint32_t> g_ExpandRacesCounter{};
    std::atomic<__uint64_t> g_SimulationCounter{};

    void ExpandTreeCPU(__uint32_t index, MctsNode *root) {
        MctsNode *node = SelectNode(root);

        /* We selected node already extended but without any children roll back */
        if (node->HasChildrenAssigned()) {
            PropagateResult(node,
                            ported_translation::IsCheck(node->m_board) ?
                            SwapColor(node->m_board.MovingColor) : DRAW
            );
            g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
            return;
        }

        MctsNode *expandedNode = ExpandNode(node);

        /* Selected node was not expanded yet, but we found out that it is a dead end indeed */
        if (expandedNode == nullptr) {
            PropagateResult(node,
                            ported_translation::IsCheck(node->m_board) ?
                            SwapColor(node->m_board.MovingColor) : DRAW
            );
            g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
            return;
        }

        const __uint32_t result = cpu::SimulateGame(index, expandedNode->m_board.DumpToExternal());
        PropagateResult(expandedNode, result);

        g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
    }

    MctsNode *SelectNode(MctsNode *const root) {
        assert(root != nullptr && "NULLPTR NODE DETECTED!");

        root->IncNumSamples();

        /* if we have no children return ourselves and let expand fail */
        /* NOTE: after selection there should be check if return node has children or not! */
        if (!root->HasChildrenAssigned() || root->GetChildren().empty()) {
            return root;
        }

        double bestScore = std::numeric_limits<double>::min();
        MctsNode *node{};

        for (const auto child: root->GetChildren()) {
            if (const double score = child->CalculateUCB(); score > bestScore) {
                bestScore = score;
                node = child;
            }
        }

        return SelectNode(node);
    }

    MctsNode *ExpandNode(MctsNode *const root) {
        if (root->GetNumSamples() < MIN_SAMPLES_TO_EXPAND) {
            return root;
        }

        /* generate all possible children */
        const auto moves = ported_translation::GenMoves(root->m_board);

        /* Alloc the bector */
        const auto pChildren = new std::vector<MctsNode *>();
        pChildren->reserve(moves.size());

        /* Add child nodes */
        for (const auto move: moves) {
            pChildren->emplace_back(new MctsNode(root, move));
        }

        const bool result = root->SetChildren(pChildren);

        if (!result) {
            g_ExpandRacesCounter.fetch_add(1, std::memory_order_relaxed);

            for (const auto &child: *pChildren) {
                delete child;
            }
            delete pChildren;
        }

        return root->GetChildren().empty() ? nullptr :
               root->GetChildren()[root->GetNumSamples() % root->GetChildren().size()];
    }

    void PropagateResult(MctsNode *const node, const __uint32_t result) {
        if (node == nullptr) {
            return;
        }

        node->ScoreNode(result);
        PropagateResult(node->m_parent, result);
    }

    results_t<EVAL_SPLIT_KERNEL_BOARDS>
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

    results_t<EVAL_PLAIN_KERNEL_BOARDS>
    SimulatePlain(const cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS> &boards, cudaStream_t &stream) {
        results_t<EVAL_PLAIN_KERNEL_BOARDS> hResults{};

        __uint32_t *dSeeds{};
        __uint32_t *dResults{};
        cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS> *dBoards{};
        BYTE *dBytes{};

        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dBoards, sizeof(cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS>), stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dSeeds, sizeof(__uint32_t) * EVAL_PLAIN_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dResults, sizeof(__uint32_t) * EVAL_PLAIN_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dBytes, sizeof(cuda_Move) * DEFAULT_STACK_SIZE, stream));
        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dBoards, &boards, sizeof(cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS>),
                                            cudaMemcpyHostToDevice, stream));

        const auto seeds = GenSeeds(EVAL_PLAIN_KERNEL_BOARDS);
        assert(seeds.size() == EVAL_PLAIN_KERNEL_BOARDS);

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dSeeds, seeds.data(), sizeof(__uint32_t) * EVAL_PLAIN_KERNEL_BOARDS,
                                            cudaMemcpyHostToDevice, stream));

        EvaluateBoardsPlainKernel<EVAL_PLAIN_KERNEL_BOARDS><<<1, EVAL_PLAIN_KERNEL_BOARDS, 0, stream>>>(
                dBoards, dSeeds, dResults, MAX_SIMULATION_DEPTH, dBytes
        );

        CUDA_ASSERT_SUCCESS(cudaGetLastError());

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(hResults.data(), dResults,
                                            sizeof(__uint32_t) * EVAL_SPLIT_KERNEL_BOARDS, cudaMemcpyDeviceToHost,
                                            stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dSeeds, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBoards, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dResults, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBytes, stream));

        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));

        return hResults;
    }
}
