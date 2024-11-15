//
// Created by Jlisowskyy on 15/11/24.
//

#ifndef SRC_HELPERS_CUH
#define SRC_HELPERS_CUH

#include <cuda_runtime.h>

void AssertSuccess(cudaError_t error, const char *file, int line);

bool TraceError(cudaError_t error, const char *file, int line);

#define CUDA_ASSERT_SUCCESS(err) AssertSuccess(err, __FILE__, __LINE__)
#define CUDA_TRACE_ERROR(err) TraceError(err, __FILE__, __LINE__)

#endif //SRC_HELPERS_CUH
