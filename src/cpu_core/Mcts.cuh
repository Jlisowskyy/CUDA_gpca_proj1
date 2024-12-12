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

static constexpr uint32_t MIN_SAMPLES_TO_EXPAND = 8;
static constexpr uint32_t MAX_SIMULATION_DEPTH = 100;

enum class EngineType {
    CPU,
    GPU0,
    GPU1,
};

namespace mcts {
    template<uint32_t BATCH_SIZE>
    using results_t = std::array<uint32_t, BATCH_SIZE>;

    template<uint32_t BATCH_SIZE>
    using sim_t = results_t<BATCH_SIZE> (*)(const cuda_PackedBoard<BATCH_SIZE> &, cudaStream_t &);

    extern std::atomic<uint32_t> g_ExpandRacesCounter;
    extern std::atomic<uint64_t> g_SimulationCounter;
    extern std::atomic<double> g_CopyTimes;
    extern std::atomic<double> g_KernelTime;
    extern std::atomic<double> g_CopyBackTimes;

    void ExpandTreeCPU(MctsNode *root);

    [[nodiscard]] MctsNode *SelectNode(MctsNode *root);

    [[nodiscard]] MctsNode *ExpandNode(MctsNode *root);

    void PropagateResult(MctsNode *node, uint32_t result);

    template<bool USE_TIMERS = false>
    results_t<EVAL_SPLIT_KERNEL_BOARDS>
    SimulateSplit(const cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS> &boards, cudaStream_t &stream) {
        results_t<EVAL_SPLIT_KERNEL_BOARDS> hResults{};

        /* Memory preprocessing */

        [[maybe_unused]] float memcpyTime;
        [[maybe_unused]] cudaEvent_t memcpyStart, memcpyStop;

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyStart));
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyStop));
            CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyStart, stream));
        }

        uint32_t *dSeeds{};
        uint32_t *dResults{};
        cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS> *dBoards{};

        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dBoards, sizeof(cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS>), stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dSeeds, sizeof(uint32_t) * EVAL_SPLIT_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dResults, sizeof(uint32_t) * EVAL_SPLIT_KERNEL_BOARDS, stream));


        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dBoards, &boards, sizeof(cuda_PackedBoard<EVAL_SPLIT_KERNEL_BOARDS>),
            cudaMemcpyHostToDevice, stream));

        const auto seeds = GenSeeds(EVAL_SPLIT_KERNEL_BOARDS);
        assert(seeds.size() == EVAL_SPLIT_KERNEL_BOARDS);

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dSeeds, seeds.data(), sizeof(uint32_t) * EVAL_SPLIT_KERNEL_BOARDS,
            cudaMemcpyHostToDevice, stream));

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyStop, stream));
            CUDA_ASSERT_SUCCESS(cudaEventSynchronize(memcpyStop));
            CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&memcpyTime, memcpyStart, memcpyStop));
        }

        /* kernel run */
        [[maybe_unused]] float kernelTime;
        [[maybe_unused]] cudaEvent_t kernelStart, kernelStop;

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&kernelStart));
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&kernelStop));
            CUDA_ASSERT_SUCCESS(cudaEventRecord(kernelStart, stream));
        }

        EvaluateBoardsSplitKernel<<<EVAL_SPLIT_KERNEL_BOARDS / WARP_SIZE, WARP_SIZE *
                                                                          BIT_BOARDS_PER_COLOR, 0, stream>>>(
            dBoards, dSeeds, dResults, MAX_SIMULATION_DEPTH, nullptr
        );

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventRecord(kernelStop, stream));
            CUDA_ASSERT_SUCCESS(cudaEventSynchronize(kernelStop));
            CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));
        }

        CUDA_ASSERT_SUCCESS(cudaGetLastError());

        /* memory operations cleanup operations */
        [[maybe_unused]] float memcpyBackTime;
        [[maybe_unused]] cudaEvent_t memcpyBackStart, memcpyBackStop;

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyBackStart));
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyBackStop));
            CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyBackStart, stream));
        }

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(hResults.data(), dResults,
            sizeof(uint32_t) * EVAL_SPLIT_KERNEL_BOARDS, cudaMemcpyDeviceToHost,
            stream));

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyBackStop, stream));
            CUDA_ASSERT_SUCCESS(cudaEventSynchronize(memcpyBackStop));
            CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&memcpyBackTime, memcpyBackStart, memcpyBackStop));
        }

        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dSeeds, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBoards, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dResults, stream));

        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));

        /* cleanup events */

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyStart));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyStop));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(kernelStart));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(kernelStop));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyBackStart));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyBackStop));

            g_KernelTime.fetch_add(double(kernelTime));
            g_CopyTimes.fetch_add(double(memcpyTime));
            g_CopyBackTimes.fetch_add(double(memcpyBackTime));
        }

        return hResults;
    }

    template<bool USE_TIMERS = false>
    results_t<EVAL_PLAIN_KERNEL_BOARDS>
    SimulatePlain(const cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS> &boards, cudaStream_t &stream) {
        results_t<EVAL_PLAIN_KERNEL_BOARDS> hResults{};

        /* Memory preprocessing */
        [[maybe_unused]] float memcpyTime;
        [[maybe_unused]] cudaEvent_t memcpyStart, memcpyStop;

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyStart));
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyStop));
            CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyStart, stream));
        }

        uint32_t *dSeeds{};
        uint32_t *dResults{};
        cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS> *dBoards{};
        BYTE *dBytes{};

        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dBoards, sizeof(cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS>), stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dSeeds, sizeof(uint32_t) * EVAL_PLAIN_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dResults, sizeof(uint32_t) * EVAL_PLAIN_KERNEL_BOARDS, stream));
        CUDA_ASSERT_SUCCESS(
            cudaMallocAsync(&dBytes, sizeof(cuda_Move) * DEFAULT_STACK_SIZE * EVAL_PLAIN_KERNEL_BOARDS, stream));

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dBoards, &boards, sizeof(cuda_PackedBoard<EVAL_PLAIN_KERNEL_BOARDS>),
            cudaMemcpyHostToDevice, stream));

        const auto seeds = GenSeeds(EVAL_PLAIN_KERNEL_BOARDS);
        assert(seeds.size() == EVAL_PLAIN_KERNEL_BOARDS);

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dSeeds, seeds.data(), sizeof(uint32_t) * EVAL_PLAIN_KERNEL_BOARDS,
            cudaMemcpyHostToDevice, stream));

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyStop, stream));
            CUDA_ASSERT_SUCCESS(cudaEventSynchronize(memcpyStop));
            CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&memcpyTime, memcpyStart, memcpyStop));
        }

        /* kernel run */
        [[maybe_unused]] float kernelTime;
        [[maybe_unused]] cudaEvent_t kernelStart, kernelStop;

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&kernelStart));
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&kernelStop));
            CUDA_ASSERT_SUCCESS(cudaEventRecord(kernelStart, stream));
        }

        EvaluateBoardsPlainKernel<EVAL_PLAIN_KERNEL_BOARDS><<<1, EVAL_PLAIN_KERNEL_BOARDS, 0, stream>>>(
            dBoards, dSeeds, dResults, MAX_SIMULATION_DEPTH, dBytes
        );

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventRecord(kernelStop, stream));
            CUDA_ASSERT_SUCCESS(cudaEventSynchronize(kernelStop));
            CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));
        }

        CUDA_ASSERT_SUCCESS(cudaGetLastError());

        /* memory operations cleanup operations */
        [[maybe_unused]] float memcpyBackTime;
        [[maybe_unused]] cudaEvent_t memcpyBackStart, memcpyBackStop;

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyBackStart));
            CUDA_ASSERT_SUCCESS(cudaEventCreate(&memcpyBackStop));
            CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyBackStart, stream));
        }

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(hResults.data(), dResults,
            sizeof(uint32_t) * EVAL_PLAIN_KERNEL_BOARDS, cudaMemcpyDeviceToHost,
            stream));

        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventRecord(memcpyBackStop, stream));
            CUDA_ASSERT_SUCCESS(cudaEventSynchronize(memcpyBackStop));
            CUDA_ASSERT_SUCCESS(cudaEventElapsedTime(&memcpyBackTime, memcpyBackStart, memcpyBackStop));
        }

        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dSeeds, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBoards, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dResults, stream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(dBytes, stream));

        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(stream));

        /* cleanup events */
        if constexpr (USE_TIMERS) {
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyStart));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyStop));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(kernelStart));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(kernelStop));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyBackStart));
            CUDA_ASSERT_SUCCESS(cudaEventDestroy(memcpyBackStop));

            g_KernelTime.fetch_add(double(kernelTime));
            g_CopyTimes.fetch_add(double(memcpyTime));
            g_CopyBackTimes.fetch_add(double(memcpyBackTime));
        }

        return hResults;
    }

    template<EngineType ENGINE_TYPE, bool USE_TIMERS = false>
    void ExpandTreeGPU(MctsNode *root, cudaStream_t &stream, const volatile bool& workCond) {
        static_assert(ENGINE_TYPE == EngineType::GPU1 || ENGINE_TYPE == EngineType::GPU0);
        static constexpr uint32_t BATCH_SIZE = ENGINE_TYPE == EngineType::GPU0
                                                   ? EVAL_SPLIT_KERNEL_BOARDS
                                                   : EVAL_PLAIN_KERNEL_BOARDS;

        cuda_PackedBoard<BATCH_SIZE> batch;
        std::array<MctsNode *, BATCH_SIZE> selectedNodes;

        for (uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            MctsNode *node{};

            uint32_t numRetries{};
            while (!node) {
                if (!workCond && numRetries > 64) {
                    return;
                }

                node = SelectNode(root);

                /* We selected node already extended but without any children roll back */
                if (node->HasChildrenAssigned()) {
                    PropagateResult(node,
                                    ported_translation::IsCheck(node->m_board)
                                        ? SwapColor(node->m_board.MovingColor)
                                        : DRAW
                    );
                    node = nullptr;
                    g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
                    ++numRetries;
                    continue;
                }

                MctsNode *expandedNode = ExpandNode(node);

                if (!expandedNode) {
                    /* Selected node was not expanded yet, but we found out that it is a dead end indeed */
                    PropagateResult(node,
                                    ported_translation::IsCheck(node->m_board)
                                        ? SwapColor(node->m_board.MovingColor)
                                        : DRAW
                    );
                    node = nullptr;
                    g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
                    ++numRetries;
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
            results = SimulateSplit<USE_TIMERS>(batch, stream);
            g_SimulationCounter.fetch_add(EVAL_SPLIT_KERNEL_BOARDS, std::memory_order::relaxed);
        } else {
            results = SimulatePlain<USE_TIMERS>(batch, stream);
            g_SimulationCounter.fetch_add(EVAL_PLAIN_KERNEL_BOARDS, std::memory_order::relaxed);
        }

        assert(results.size() == selectedNodes.size());
        for (uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            PropagateResult(selectedNodes[idx], results[idx]);
        }
    }
}

#endif //SRC_MCTS_CUH
