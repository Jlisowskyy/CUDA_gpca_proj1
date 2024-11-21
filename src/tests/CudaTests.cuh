//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_CUDATESTS_CUH
#define SRC_CUDATESTS_CUH

#include <unordered_map>
#include <string>
#include <tuple>

/* defines */
struct cudaDeviceProp;

using TestFunc = void (*)(int, const cudaDeviceProp &deviceProps);

/* test funcs */
void FancyMagicTest(int threadsAvailable, const cudaDeviceProp &deviceProps);

void MoveGenTest(int threadsAvailable, const cudaDeviceProp &deviceProps);

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
                        "Tests the performance of the move generation on the GPU",
                        &MoveGenTest
                )
        }
};

std::tuple<unsigned, unsigned> GetDims(int threadsAvailable, const cudaDeviceProp &deviceProps);

#endif //SRC_CUDATESTS_CUH