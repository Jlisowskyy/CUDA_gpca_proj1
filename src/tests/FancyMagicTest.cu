//
// Created by Jlisowskyy on 16/11/24.
//

#include "CudaTests.cuh"

#include <cuda_runtime.h>

#include <iostream>

void FancyMagicTest(int blockSize, const cudaDeviceProp &deviceProps) {
    std::cout << "Fancy Magic Test" << std::endl;
}