//
// Created by Jlisowskyy on 13/12/24.
//

#include "TextureRookMap.cuh"

__device__ cudaTextureObject_t G_TEXTURE_ROOK_MAP;
__device__ MapAccessor G_MAP_ACCESSOR;
RookMapTexture G_ROOK_MAP_TEXTURE_OBJ{};
