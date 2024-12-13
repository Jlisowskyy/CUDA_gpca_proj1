//
// Created by Jlisowskyy on 13/12/24.
//

#ifndef TEXTUREROOKMAP_CUH
#define TEXTUREROOKMAP_CUH

#include "FancyMagicRookMap.cuh"
#include "cuda_PackedBoard.cuh"
#include "cuda_Array.cuh"

#include <cuda_runtime.h>

class MapAccessor;
class RookMapTexture;

extern __device__ cudaTextureObject_t G_TEXTURE_ROOK_MAP;
extern __device__ MapAccessor G_MAP_ACCESSOR;
extern RookMapTexture G_ROOK_MAP_TEXTURE_OBJ;

class MapAccessor final {
    static constexpr uint64_t MASK32 = 0xFFFFFFFF;
    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------
public:
    MapAccessor() = default;

    ~MapAccessor() = default;

    MapAccessor(const cuda_Array<uint64_t, BIT_BOARD_FIELDS> &magics,
                const cuda_Array<uint64_t, BIT_BOARD_FIELDS> &offsets) {
        auto simpleMap = FancyMagicRookMap::Create().GetMaps();

        for (uint32_t idx = 0; idx < BIT_BOARD_FIELDS; ++idx) {
            const uint32_t hiMagic = uint32_t(magics[idx] >> 32);
            const uint32_t loMagic = uint32_t(magics[idx] & MASK32);

            m_magics[idx] = hiMagic;
            m_magics[idx + BIT_BOARD_FIELDS] = loMagic;
            m_offsets[idx] = 64u - uint32_t(offsets[idx]);

            const uint64_t mask = simpleMap[idx].getFullMask();
            m_masks[idx] = uint32_t(mask >> 32);
            m_masks[idx + BIT_BOARD_FIELDS] = uint32_t(mask & MASK32);
        }
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_CALL_ALWAYS constexpr uint64_t GetMagic(const uint32_t idx) const {
        const uint32_t hi = m_magics[idx];
        const uint32_t lo = m_magics[idx + BIT_BOARD_FIELDS];
        return uint64_t(hi) << 32 | lo;
    }

    [[nodiscard]] FAST_CALL_ALWAYS constexpr uint64_t GetOffset(const uint32_t idx) const {
        return m_offsets[idx];
    }

    [[nodiscard]] FAST_CALL_ALWAYS constexpr uint64_t GetMask(const uint32_t idx) const {
        const uint32_t hi = m_masks[idx];
        const uint32_t lo = m_masks[idx + BIT_BOARD_FIELDS];
        return uint64_t(hi) << 32 | lo;
    }

    [[nodiscard]] FAST_CALL_ALWAYS constexpr uint64_t GetIdx(const uint32_t msbInd, const uint64_t fullBoard) const {
        const uint64_t neighbors = fullBoard & GetMask(msbInd);
        const uint64_t magic = GetMagic(msbInd);
        const uint64_t offset = GetOffset(msbInd);

        return (neighbors * magic) >> offset;
    }

    // ------------------------------
    // Class fields
    // ------------------------------
private:
    alignas(128) cuda_Array<uint32_t, BIT_BOARD_FIELDS * 2> m_magics;
    alignas(128) cuda_Array<uint32_t, BIT_BOARD_FIELDS * 2> m_masks;
    alignas(128) cuda_Array<uint32_t, BIT_BOARD_FIELDS> m_offsets;
};

class RookMapTexture final {
    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------
public:
    RookMapTexture() {
        auto simpleMap = FancyMagicRookMap::Create().GetMaps();

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint64_t>();
        CUDA_ASSERT_SUCCESS(cudaMallocArray(&m_cuArray, &channelDesc, 64, 4096));

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_cuArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;

        for (uint32_t idx = 0; idx < BIT_BOARD_FIELDS; ++idx) {
            CUDA_ASSERT_SUCCESS(cudaMemcpy2DToArray(m_cuArray, 0, idx, simpleMap[idx].data(),
                sizeof(uint64_t), sizeof(uint64_t), 4096,
                cudaMemcpyHostToDevice));
        }

        CUDA_ASSERT_SUCCESS(cudaCreateTextureObject(&m_tex, &resDesc, &texDesc, NULL));
    }

    ~RookMapTexture() {
        CUDA_ASSERT_SUCCESS(cudaFreeArray(m_cuArray));
        CUDA_ASSERT_SUCCESS(cudaDestroyTextureObject(m_tex));
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] cudaTextureObject_t GetTexture() const {
        return m_tex;
    }

    // ------------------------------
    // Class fields
    // ------------------------------
private:
    cudaArray *m_cuArray;
    cudaTextureObject_t m_tex;
};

inline void InitRookMapTexture() {
    const MapAccessor accessor{MAGICS_ROOK_PARAMS, OFFSETS_ROOK_PARAMS};
    const cudaTextureObject_t tex = G_ROOK_MAP_TEXTURE_OBJ.GetTexture();

    /* copy to cuda symbol */
    CUDA_ASSERT_SUCCESS(cudaMemcpyToSymbol(G_TEXTURE_ROOK_MAP, &tex, sizeof(cudaTextureObject_t), 0,
        cudaMemcpyHostToDevice));
    CUDA_ASSERT_SUCCESS(cudaMemcpyToSymbol(G_MAP_ACCESSOR, &accessor, sizeof(MapAccessor), 0, cudaMemcpyHostToDevice));
}

class TextureRookMap final {
    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------
public:
    TextureRookMap() = delete;

    ~TextureRookMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint32_t GetBoardIndex(const uint32_t color) {
        return BIT_BOARDS_PER_COLOR * color + ROOK_INDEX;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static uint64_t
    GetMoves(const uint32_t msbInd, const uint64_t fullBoard, [[maybe_unused]] uint64_t = 0) {
        const uint64_t idx = G_MAP_ACCESSOR.GetIdx(msbInd, fullBoard);
        return tex2D<uint64_t>(G_TEXTURE_ROOK_MAP, float(msbInd), float(idx));
    }

    template<uint32_t NUM_BOARDS>
    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint32_t
    GetMatchingCastlingIndex(const typename cuda_PackedBoard<NUM_BOARDS>::BoardFetcher &fetcher, uint64_t figBoard) {
        uint32_t rv{};

        uint32_t index = fetcher.MovingColor() * CASTLINGS_PER_COLOR;
        rv += (1 + index) * (fetcher.GetCastlingRight(index) && ((CASTLING_ROOK_MAPS[index] & figBoard) != 0));

        index += 1;
        rv += (1 + index) * (fetcher.GetCastlingRight(index) && ((CASTLING_ROOK_MAPS[index] & figBoard) != 0));

        return rv == 0 ? SENTINEL_CASTLING_INDEX : rv - 1;
    }
};

#endif //TEXTUREROOKMAP_CUH
