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

/* TODO: move to CMAKE */
#define AGGRESSIVE_INLINE

#ifdef AGGRESSIVE_INLINE

#define INLINE __forceinline__

#else

#define INLINE inline

#endif // AGGRESSIVE_INLINE

#define FAST_CALL INLINE HYBRID

#define FAST_DCALL INLINE __device__

#define FAST_DCALL_ALWAYS __forceinline__ __device__

template<class NumT>
FAST_DCALL_ALWAYS void simpleRand(NumT &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
}

#endif //SRC_HELPERS_CUH
