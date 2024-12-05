//
// Created by Jlisowskyy on 05/12/24.
//

#ifndef SRC_MCTS_CUH
#define SRC_MCTS_CUH

#include "MctsNode.cuh"
#include "../cuda_core/cuda_PackedBoard.cuh"
#include "../tests/CudaTests.cuh"

#include "../cuda_core/ComputeKernels.cuh"

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

    extern std::atomic<__uint32_t> ExpandRacesCounter;
    extern std::atomic<__uint64_t> SimulationCounter;

    void ExpandTreeCPU(MctsNode *root);

    [[nodiscard]] MctsNode *SelectNode(MctsNode *root);

    [[nodiscard]] MctsNode *ExpandNode(MctsNode *root);

    void PropagateResult(MctsNode *node, __uint32_t result);

    template<__uint32_t BATCH_SIZE>
    results_t<BATCH_SIZE> SimulateSplit(const cuda_PackedBoard<BATCH_SIZE> &boards, cudaStream_t &stream) {
        results_t<BATCH_SIZE> hResults{};

        __uint32_t *dSeeds{};
        __uint32_t *dResults{};
        cuda_PackedBoard<BATCH_SIZE> *dBoards{};

        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dBoards, sizeof(cuda_PackedBoard<BATCH_SIZE>), stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dSeeds, sizeof(__uint32_t) * BATCH_SIZE, stream));
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&dResults, sizeof(__uint32_t) * BATCH_SIZE, stream));
        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dBoards, &boards,
                                            sizeof(cuda_PackedBoard<BATCH_SIZE>), cudaMemcpyHostToDevice, stream));

        const auto seeds = GenSeeds(BATCH_SIZE);

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(dSeeds, seeds.data(),
                                            sizeof(__uint32_t) * BATCH_SIZE, cudaMemcpyHostToDevice, stream));

        EvaluateBoardsSplitKernel<BATCH_SIZE><<<1, BATCH_SIZE * BIT_BOARDS_PER_COLOR>>>(
                dBoards, dSeeds, dResults, MAX_SIMULATION_DEPTH, nullptr
        );
        CUDA_ASSERT_SUCCESS(cudaGetLastError());

        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(hResults.data(), dResults,
                                            sizeof(__uint32_t) * BATCH_SIZE, cudaMemcpyDeviceToHost, stream));
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

    template<__uint32_t BATCH_SIZE, sim_t<BATCH_SIZE> FUNC = SimulateSplit<BATCH_SIZE>>
    void ExpandTreeGPU(MctsNode *root, cudaStream_t &stream) {
        cuda_PackedBoard<BATCH_SIZE> batch;
        std::array<MctsNode *, BATCH_SIZE> selectedNodes;

        for (__uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            auto node = SelectNode(root);
            node = ExpandNode(node);

            selectedNodes[idx] = node;
            batch.saveBoard(idx, node->m_board);
        }

        const auto results = FUNC(batch, stream);

        assert(results.size() == selectedNodes.size());
        for (__uint32_t idx = 0; idx < BATCH_SIZE; ++idx) {
            PropagateResult(selectedNodes[idx], results[idx]);
        }
    }


}

#endif //SRC_MCTS_CUH
