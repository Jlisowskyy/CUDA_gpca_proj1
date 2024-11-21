//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_CUDATESTS_CUH
#define SRC_CUDATESTS_CUH

#include "../cuda_core/Helpers.cuh"

#include <unordered_map>
#include <string>
#include <tuple>

/* defines */
struct cudaDeviceProp;

using TestFunc = void (*)(__uint32_t, const cudaDeviceProp &deviceProps);

/* test funcs */
void FancyMagicTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

void MoveGenTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

void MoveGenPerfTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

template<class NumT>
FAST_DCALL void simpleRand(NumT &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
}

/* test mapping */
static const std::unordered_map<std::string, std::tuple<std::string, std::string, TestFunc>> CudaTestsMap = {
        {
                "magic_test",
                std::make_tuple(
                        "Fancy Magic Test",
                        "Measures average access times of Fancy mappings on the GPU as well as correctness",
                        &FancyMagicTest
                )
        },
        {
                "move_gen",
                std::make_tuple(
                        "MoveGen Test",
                        "Tests the correctness of the move generation on the GPU",
                        &MoveGenTest
                )
        },
        {
                "move_perf",
                std::make_tuple(
                        "MoveGen Performance Test",
                        "Tests the performance of the move generation on the GPU",
                        &MoveGenPerfTest
                )
        },
};

std::tuple<__uint32_t, __uint32_t> GetDims(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

#endif //SRC_CUDATESTS_CUH