//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_CUDATESTS_CUH
#define SRC_CUDATESTS_CUH

#include "../cuda_core/Helpers.cuh"

#include <unordered_map>
#include <string>
#include <tuple>
#include <vector>

/* defines */
struct cudaDeviceProp;

using TestFunc = void (*)(__uint32_t, const cudaDeviceProp &deviceProps);

/* test funcs */
void FancyMagicTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

void MoveGenTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

void MoveGenPerfTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

inline void MoveGenCorPerfTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    MoveGenTest(threadsAvailable, deviceProps);
    MoveGenPerfTest(threadsAvailable, deviceProps);
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
        {
                "move_gen_perf",
                std::make_tuple(
                        "MoveGen PerfCor Test",
                        "Tests first the correctness of the move generation and performance after it",
                        &MoveGenCorPerfTest
                )
        },
};

/* Helpers */
std::tuple<__uint32_t, __uint32_t> GetDims(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

std::vector<std::string> LoadFenDb();

std::vector<__uint32_t> GenSeeds(__uint32_t size);

void PolluteCache();

#endif //SRC_CUDATESTS_CUH