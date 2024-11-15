//
// Created by Jlisowskyy on 15/11/24.
//

#include "Helpers.cuh"

#include <iostream>
#include <format>

void AssertSuccess(cudaError_t error, const char *file, int line) {
    TraceError(error, file, line);

    if (error != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
}

bool TraceError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        std::cerr << std::format("CUDA Error at {}:{} - {}\n", file, line, cudaGetErrorString(error)) << std::endl;
    }

    return error != cudaSuccess;
}
