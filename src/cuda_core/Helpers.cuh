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

#define HYBRID __host__ __device__

template<typename T>
HYBRID constexpr bool less_equal_comp(const T &a, const T &b) {
    return a <= b;
}

template<typename T>
HYBRID constexpr bool greater_equal_comp(const T &a, const T &b) {
    return a >= b;
}

template<typename T>
HYBRID constexpr bool less_comp(const T &a, const T &b) {
    return a < b;
}

template<typename T>
HYBRID constexpr bool greater_comp(const T &a, const T &b) {
    return a > b;
}

template<typename T>
HYBRID constexpr T cuda_min(T a, T b) {
    return a < b ? a : b;
}

template<typename T>
HYBRID constexpr T cuda_max(T a, T b) {
    return a > b ? a : b;
}

#endif //SRC_HELPERS_CUH
