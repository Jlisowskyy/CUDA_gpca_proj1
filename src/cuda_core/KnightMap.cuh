//
// Created by Jlisowskyy on 12/28/23.
//

#ifndef KNIGHTMAP_H
#define KNIGHTMAP_H

#include <cuda_runtime.h>

#include "MoveGenerationUtils.cuh"
#include "Helpers.cuh"
#include "cuda_Array.cuh"

namespace KnightMapConstants {
    __device__ static constexpr __uint32_t maxMovesCount = 8;

    // Describes knight possible moves coordinates.
    __device__ static constexpr int movesCords[] = {6, 15, 17, 10, -6, -15, -17, -10};

    // Accordingly describes y positions after the move relatively to knight's y position.
    // Used to omit errors during generation.
    __device__ static constexpr int rowCords[] = {1, 2, 2, 1, -1, -2, -2, -1};

    alignas(128) __device__ static constexpr cuda_Array<__uint64_t, BIT_BOARD_FIELDS> movesMap =
            GenStaticMoves(maxMovesCount, movesCords, rowCords);

    extern __device__ cudaTextureObject_t deviceTextureObj;

    struct TextureResources {
        cudaTextureObject_t textureObj{};
        cudaArray *deviceArray{};

        TextureResources() = default;

        void Initialize() {
            static constexpr __uint64_t MASK = 0xFFFFFFFF;

            cuda_Array<__uint64_t, BIT_BOARD_FIELDS> hostMovesMap =
                    GenStaticMoves(maxMovesCount, movesCords, rowCords);

            // Create uint64_t to int2 conversion array
            auto *int2Data = new int2[BIT_BOARD_FIELDS];
            for (size_t i = 0; i < BIT_BOARD_FIELDS; ++i) {
                int2Data[i].x = (int) (hostMovesMap[i] & MASK);
                int2Data[i].y = (int) ((hostMovesMap[i] >> 32) & MASK);
            }

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();

            cudaExtent extent = make_cudaExtent(BIT_BOARD_FIELDS, 1, 0);
            CUDA_ASSERT_SUCCESS(cudaMalloc3DArray(&deviceArray, &channelDesc, extent));

            cudaMemcpy3DParms copyParams{};
            copyParams.srcPtr = make_cudaPitchedPtr(int2Data,
                                                    BIT_BOARD_FIELDS * sizeof(int2),
                                                    BIT_BOARD_FIELDS, 1);
            copyParams.dstArray = deviceArray;
            copyParams.extent = extent;
            copyParams.kind = cudaMemcpyHostToDevice;

            CUDA_ASSERT_SUCCESS(cudaMemcpy3D(&copyParams));
            delete[] int2Data;

            cudaTextureDesc texDesc{};
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaResourceDesc resDesc{};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = deviceArray;

            CUDA_ASSERT_SUCCESS(cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr));

            CUDA_ASSERT_SUCCESS(cudaMemcpyToSymbol(deviceTextureObj, &textureObj, sizeof(cudaTextureObject_t)));
        }

        void Cleanup() {
            if (textureObj) {
                CUDA_ASSERT_SUCCESS(cudaDestroyTextureObject(textureObj));
                textureObj = 0;
            }
            if (deviceArray) {
                CUDA_ASSERT_SUCCESS(cudaFreeArray(deviceArray));
                deviceArray = nullptr;
            }
        }

        ~TextureResources() {
            Cleanup();
        }
    };

    extern TextureResources textureMap;
}

class KnightMap final {
    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------

public:
    KnightMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t GetBoardIndex(int color) {
        return BIT_BOARDS_PER_COLOR * color + KNIGHT_INDEX;
    }

//    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t
//    GetMoves(__uint32_t msbInd, [[maybe_unused]] __uint64_t = 0,
//             [[maybe_unused]] __uint64_t = 0) { return KnightMapConstants::movesMap[msbInd]; }

    static __forceinline__ __device__ __uint64_t int2_as_longlong(int2 a) {
        __uint64_t lo = a.x;
        __uint64_t hi = a.y;

        return lo | (hi << 32);
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static __uint64_t
    GetMoves(__uint32_t msbInd, [[maybe_unused]] __uint64_t = 0, [[maybe_unused]] __uint64_t = 0) {
        int2 v = tex2D<int2>(KnightMapConstants::deviceTextureObj, msbInd, 0);
        return int2_as_longlong(v);
    }

};


#endif // KNIGHTMAP_H
