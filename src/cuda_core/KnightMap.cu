//
// Created by Jlisowskyy on 01/12/24.
//

#include "KnightMap.cuh"

KnightMapConstants::TextureResources KnightMapConstants::textureMap{};

__device__ cudaTextureObject_t KnightMapConstants::deviceTextureObj{};
