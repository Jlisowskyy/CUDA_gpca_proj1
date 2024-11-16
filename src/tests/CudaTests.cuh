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
void FancyMagicTest(int blockSize, const cudaDeviceProp &deviceProps);

/* test mapping */
static const std::unordered_map<std::string, std::tuple<std::string, std::string, TestFunc>> CudaTestsMap = {
        {
                "magic_test",
                std::make_tuple(
                        "Fancy Magic Test",
                        "Measures average access times of Fancy mappings on the GPU",
                        &FancyMagicTest
                )
        }
        // Add more test entries here in the same format:
        // {
        //     "TestIdentifier",
        //     std::make_tuple(
        //         "Test Display Name",
        //         "Test Description",
        //         &TestFunctionName
        //     )
        // }
};

#endif //SRC_CUDATESTS_CUH