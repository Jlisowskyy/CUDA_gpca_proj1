//
// Created by Jlisowskyy on 2/14/24.
//

#ifndef BLACKPAWNMAP_H
#define BLACKPAWNMAP_H

#include <cstdint>
#include <cuda_runtime.h>

#include "MoveGenerationUtils.cuh"
#include "Helpers.cuh"

namespace BlackPawnMapConstants {
    __device__ __constant__ static constexpr __uint64_t PromotingMask = GenMask(8, 16, 1);

    // Mask indicating whether black pawn can be killed on that field by enemy pawn
    __device__ __constant__ static constexpr __uint64_t ElPassantMask = GenMask(32, 40, 1);

    // Mask indicating whether white pawn can be killed on that field by enemy pawn
    __device__ __constant__ static constexpr __uint64_t EnemyElPassantMask = GenMask(24, 32, 1);

    // Mask with ones only on "x7" line
    __device__ __constant__ static constexpr __uint64_t StartMask = GenMask(48, 56, 1);

    // Mask with ones only on "Ax" line
    __device__ __constant__ static constexpr __uint64_t LeftMask = ~GenMask(0, 57, 8);

    // Mask with ones only on "Hx" line
    __device__ __constant__ static constexpr __uint64_t RightMask = ~GenMask(7, 64, 8);
}

class BlackPawnMap final {
    // ------------------------------
    // Class creation
    // ------------------------------

public:
    BlackPawnMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t
    GetBoardIndex([[maybe_unused]] const int color) { return B_PAWN_INDEX; }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr int GetColor() { return BLACK; }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t GetEnemyPawnBoardIndex() { return W_PAWN_INDEX; }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetAttackFields(__uint64_t pawnBits) {
        const __uint64_t leftAttack = (BlackPawnMapConstants::LeftMask & pawnBits) >> 9;
        const __uint64_t rightAttack = (BlackPawnMapConstants::RightMask & pawnBits) >> 7;
        return leftAttack | rightAttack;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetPlainMoves(__uint64_t pawnBit, __uint64_t fullMap) {
        const __uint64_t frontMove = (pawnBit >> 8) & ~fullMap;

        const __uint64_t isOnStartField = ((frontMove << 8) & pawnBit & BlackPawnMapConstants::StartMask) >> 16;
        const __uint64_t frontDoubleMove = isOnStartField & ~fullMap;

        return frontMove | frontDoubleMove;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t
    GetSinglePlainMoves(__uint64_t pawnBit, __uint64_t fullMap) {
        return (pawnBit >> 8) & ~fullMap;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t RevertSinglePlainMoves(__uint64_t pawnBit) {
        return pawnBit << 8;
    }

    // Returns all moves excepts ElPassantOnes
    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t
    GetMoves(int msbPos, __uint64_t fullMap, __uint64_t enemyMap) {
        const __uint64_t pawnBit = cuda_MaxMsbPossible >> msbPos;
        const __uint64_t attackMoves = GetAttackFields(pawnBit) & enemyMap;
        const __uint64_t plainMoves = GetPlainMoves(pawnBit, fullMap);

        return attackMoves | plainMoves;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetElPassantSuspectedFields(__uint64_t elPassantField) {
        const __uint64_t leftField = (BlackPawnMapConstants::LeftMask & elPassantField) >> 1;
        const __uint64_t rightField = (BlackPawnMapConstants::RightMask & elPassantField) << 1;
        return leftField | rightField;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetElPassantMoveField(__uint64_t elPassantField) {
        return elPassantField >> 8;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t
    GetElPassantField(__uint64_t moveField, __uint64_t startField) {
        return moveField & BlackPawnMapConstants::ElPassantMask & (BlackPawnMapConstants::StartMask & startField) >> 16;
    }
};

#endif // BLACKPAWNMAP_H
