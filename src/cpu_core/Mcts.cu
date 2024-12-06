//
// Created by Jlisowskyy on 05/12/24.
//

#include "Mcts.cuh"

#include "cpu_MoveGen.cuh"

#include <numeric>

namespace mcts {
    std::atomic<__uint32_t> g_ExpandRacesCounter{};
    std::atomic<__uint64_t> g_SimulationCounter{};
    std::atomic<double> g_CopyTimes{};
    std::atomic<double> g_KernelTime{};
    std::atomic<double> g_CopyBackTimes{};

    void ExpandTreeCPU(MctsNode *root) {
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

        const __uint32_t result = cpu::SimulateGame(expandedNode->m_board.DumpToExternal());
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

        /* Memory preprocessing */

        float memcpyTime;
        cudaEvent_t memcpyStart, memcpyStop;
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyStart));
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyStop));
        CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyStart, stream));

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

        CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyStop, stream));
        CUDA_ASSERT_SUCCESS(cudaEventSynchronize(memcpyStop));
        CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&memcpyTime, memcpyStart, memcpyStop));

        /* kernel run */
        float kernelTime;
        cudaEvent_t kernelStart, kernelStop;
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&kernelStart));
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&kernelStop));
        CUDA_ASSERT_SUCCESS(cudaEventRecord(kernelStart, stream));

        EvaluateBoardsSplitKernel<<<EVAL_SPLIT_KERNEL_BOARDS / WARP_SIZE, WARP_SIZE *
                                                                          BIT_BOARDS_PER_COLOR, 0, stream>>>(
                dBoards, dSeeds, dResults, MAX_SIMULATION_DEPTH, nullptr
        );

        CUDA_ASSERT_SUCCESS(cudaEventRecord(kernelStop, stream));
        CUDA_ASSERT_SUCCESS(cudaEventSynchronize(kernelStop));
        CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));

        CUDA_ASSERT_SUCCESS(cudaGetLastError());

        /* memory operations cleanup operations */
        float memcpyBackTime;
        cudaEvent_t memcpyBackStart, memcpyBackStop;
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyBackStart));
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyBackStop));
        CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyBackStart, stream));

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(hResults.data(), dResults,
                                            sizeof(__uint32_t) * EVAL_SPLIT_KERNEL_BOARDS, cudaMemcpyDeviceToHost,
                                            stream));

        CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyBackStop, stream));
        CUDA_ASSERT_SUCCESS(cudaEventSynchronize(memcpyBackStop));
        CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&memcpyBackTime, memcpyBackStart, memcpyBackStop));

        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dSeeds, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBoards, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dResults, stream));

        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));

        /* cleanup events */
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyStart));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyStop));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(kernelStart));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(kernelStop));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyBackStart));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyBackStop));

        g_KernelTime.fetch_add(double(kernelTime));
        g_CopyTimes.fetch_add(double(memcpyTime));
        g_CopyBackTimes.fetch_add(double(memcpyBackTime));

        return hResults;
    }

    results_t<EVAL_PLAIN_KERNEL_BOARDS>
    SimulatePlain(const cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS> &boards, cudaStream_t &stream) {
        results_t<EVAL_PLAIN_KERNEL_BOARDS> hResults{};

        /* Memory preprocessing */
        float memcpyTime;
        cudaEvent_t memcpyStart, memcpyStop;
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyStart));
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyStop));
        CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyStart, stream));

        __uint32_t *dSeeds{};
        __uint32_t *dResults{};
        cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS> *dBoards{};
        BYTE *dBytes{};

        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dBoards, sizeof(cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS>), stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dSeeds, sizeof(__uint32_t) * EVAL_PLAIN_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dResults, sizeof(__uint32_t) * EVAL_PLAIN_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(
                cudaMallocAsync(&dBytes, sizeof(cuda_Move) * DEFAULT_STACK_SIZE * EVAL_PLAIN_KERNEL_BOARDS, stream));

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dBoards, &boards, sizeof(cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS>),
                                            cudaMemcpyHostToDevice, stream));

        const auto seeds = GenSeeds(EVAL_PLAIN_KERNEL_BOARDS);
        assert(seeds.size() == EVAL_PLAIN_KERNEL_BOARDS);

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dSeeds, seeds.data(), sizeof(__uint32_t) * EVAL_PLAIN_KERNEL_BOARDS,
                                            cudaMemcpyHostToDevice, stream));

        CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyStop, stream));
        CUDA_ASSERT_SUCCESS(cudaEventSynchronize(memcpyStop));
        CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&memcpyTime, memcpyStart, memcpyStop));

        /* kernel run */
        float kernelTime;
        cudaEvent_t kernelStart, kernelStop;
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&kernelStart));
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&kernelStop));
        CUDA_ASSERT_SUCCESS(cudaEventRecord(kernelStart, stream));

        EvaluateBoardsPlainKernel<EVAL_PLAIN_KERNEL_BOARDS><<<1, EVAL_PLAIN_KERNEL_BOARDS, 0, stream>>>(
                dBoards, dSeeds, dResults, MAX_SIMULATION_DEPTH, dBytes
        );

        CUDA_ASSERT_SUCCESS(cudaEventRecord(kernelStop, stream));
        CUDA_ASSERT_SUCCESS(cudaEventSynchronize(kernelStop));
        CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));

        CUDA_ASSERT_SUCCESS(cudaGetLastError());

        /* memory operations cleanup operations */
        float memcpyBackTime;
        cudaEvent_t memcpyBackStart, memcpyBackStop;
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyBackStart));
        CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyBackStop));
        CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyBackStart, stream));

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(hResults.data(), dResults,
                                            sizeof(__uint32_t) * EVAL_PLAIN_KERNEL_BOARDS, cudaMemcpyDeviceToHost,
                                            stream));

        CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyBackStop, stream));
        CUDA_ASSERT_SUCCESS(cudaEventSynchronize(memcpyBackStop));
        CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&memcpyBackTime, memcpyBackStart, memcpyBackStop));

        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dSeeds, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBoards, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dResults, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBytes, stream));

        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));

        /* cleanup events */
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyStart));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyStop));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(kernelStart));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(kernelStop));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyBackStart));
        CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyBackStop));

        g_KernelTime.fetch_add(double(kernelTime));
        g_CopyTimes.fetch_add(double(memcpyTime));
        g_CopyBackTimes.fetch_add(double(memcpyBackTime));

        return hResults;
    }
}
