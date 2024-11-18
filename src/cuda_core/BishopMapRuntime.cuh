//
// Created by Jlisowskyy on 18/11/24.
//

#ifndef SRC_BISHOPMAPRUNTIME_CUH
#define SRC_BISHOPMAPRUNTIME_CUH

#include <cuda_runtime.h>

#include "Helpers.cuh"

#include "cuda_BitOperations.cuh"

class BishopMapRuntime final {
public:

    BishopMapRuntime() = delete;

    ~BishopMapRuntime() = delete;


    [[nodiscard]] FAST_CALL static __uint64_t
    GetMoves(__uint32_t msbInd, __uint64_t fullBoard, [[maybe_unused]] __uint64_t = 0) {
        const __uint64_t startPos = cuda_MaxMsbPossible >> msbInd;

        return startPos;
    }
};

#endif //SRC_BISHOPMAPRUNTIME_CUH
