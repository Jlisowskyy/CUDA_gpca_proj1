//
// Created by Jlisowskyy on 12/31/23.
//

#ifndef PAWNMAP_H
#define PAWNMAP_H

#include <cstdint>

#include "MoveGenerationUtils.cuh"

namespace WhitePawnMapConstants {
    __device__ static constexpr __uint64_t PromotingMask = GenMask(48, 56, 1);

    // Mask indicating whether enemy pawn can kill white pawn on that field
    __device__ static constexpr __uint64_t ElPassantMask = GenMask(24, 32, 1);

    // Mask indicating whether enemy pawn can kill black pawn on that field
    __device__ static constexpr __uint64_t EnemyElPassantMask = GenMask(32, 40, 1);

    // Mask with ones only on "x2" line
    __device__ static constexpr __uint64_t StartMask = GenMask(8, 16, 1);

    // Mask with ones only on "Ax" line
    __device__ static constexpr __uint64_t LeftMask = ~GenMask(0, 57, 8);

    // Mask with ones only on "Hx" line
    __device__ static constexpr __uint64_t RightMask = ~GenMask(7, 64, 8);
}

class WhitePawnMap final
{
    // ------------------------------
    // Class creation
    // ------------------------------

    public:
    WhitePawnMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t GetBoardIndex([[maybe_unused]] int color) { return W_PAWN_INDEX; }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr int GetColor() { return WHITE; }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t GetEnemyPawnBoardIndex() { return B_PAWN_INDEX; }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetAttackFields(__uint64_t pawnBits) {
        const __uint64_t leftAttack = (WhitePawnMapConstants::LeftMask & pawnBits) << 7;
        const __uint64_t rightAttack = (WhitePawnMapConstants::RightMask & pawnBits) << 9;
        return leftAttack | rightAttack;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetPlainMoves(__uint64_t pawnBit, __uint64_t fullMap) {
        const __uint64_t frontMove = (pawnBit << 8) & ~fullMap;

        const __uint64_t isOnStartField = ((frontMove >> 8) & pawnBit & WhitePawnMapConstants::StartMask) << 16;
        const __uint64_t frontDoubleMove = isOnStartField & ~fullMap;

        return frontMove | frontDoubleMove;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetSinglePlainMoves(__uint64_t pawnBit, __uint64_t fullMap) {
        return (pawnBit << 8) & ~fullMap;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t RevertSinglePlainMoves(__uint64_t pawnBit) {
        return pawnBit >> 8;
    }

    // Returns all moves excepts ElPassantOnes
    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetMoves(int msbPos, __uint64_t fullMap, __uint64_t enemyMap) {
        const __uint64_t pawnBit = cuda_MaxMsbPossible >> msbPos;
        const __uint64_t attackMoves = GetAttackFields(pawnBit) & enemyMap;
        const __uint64_t plainMoves = GetPlainMoves(pawnBit, fullMap);

        return attackMoves | plainMoves;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetElPassantSuspectedFields(__uint64_t elPassantField) {
        const __uint64_t leftField = (WhitePawnMapConstants::LeftMask & elPassantField) >> 1;
        const __uint64_t rightField = (WhitePawnMapConstants::RightMask & elPassantField) << 1;
        return leftField | rightField;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t GetElPassantMoveField(__uint64_t elPassantField) {
        return elPassantField << 8;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static __uint64_t GetElPassantField(__uint64_t moveField, __uint64_t startField) {
        return moveField & WhitePawnMapConstants::ElPassantMask & (WhitePawnMapConstants::StartMask & startField) << 16;
    }
};

#endif // PAWNMAP_H
