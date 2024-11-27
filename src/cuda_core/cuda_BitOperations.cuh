//
// Created by Jlisowskyy on 12/29/23.
//

#ifndef CUDA_BIT_OPERATIONS_H
#define CUDA_BIT_OPERATIONS_H

#include "Helpers.cuh"

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

/* This values should be used inside every shifting operation to ensure correct types under the operation */
__device__ static constexpr __uint64_t cuda_MinMsbPossible = static_cast<__uint64_t>(1);
__device__ static constexpr __uint64_t cuda_MaxMsbPossible = cuda_MinMsbPossible << 63;

/*
 *  This header collects some functions used to manipulate bits in a 64-bit unsigned integers.
 *  The functions are constexpr, so they can be used at compile time.
 *  Most of them depends on c++20 <bit> header, so they allow to perform operations on bits in a very efficient way.
 *  They should also be platform independent with respect to c++20 implementation.
 *  To ensure same style around the project only those functions should be used to manipulate bits.
 *
 * */

/* Function efficiently computes MsbPos */
FAST_DCALL_ALWAYS int ExtractMsbPos(const __uint64_t x) { return __clzll(static_cast<int64_t>(x)); }

HYBRID constexpr int ExtractMsbPosConstexpr(const __uint64_t x) {
    if (x == 0) {
        return 0;
    }

    __uint64_t cursor = cuda_MaxMsbPossible;
    int count = 0;

    while ((cursor & x) == 0) {
        cursor >>= 1;
        ++count;
    }

    return count;
}

/* Nice way to reverse from MsbPos to LsbPos and other way around */
FAST_CALL constexpr __uint32_t ConvertToReversedPos(const __uint32_t x) {
    return x ^ 63; // equals to 63 - x;
}

template<typename NumT>
FAST_CALL constexpr void SetBitBoardBit(NumT& bitBoard, const __uint32_t pos, const bool value) {
    bitBoard = (value << pos) | (bitBoard & ~(cuda_MinMsbPossible << pos));
}

FAST_CALL constexpr __uint32_t SwapColor(const __uint32_t col) { return col ^ 1; }

/* Simply Runs 'ExtractMsbPos' and applies 'ConvertToReversedPos' on it */
FAST_DCALL_ALWAYS int ExtractMsbReversedPos(const __uint64_t x) { return ConvertToReversedPos(ExtractMsbPos(x)); }

/* Function efficiently computes LsbPos */
FAST_DCALL_ALWAYS int ExtractLsbPos(const __uint64_t x) { return __ffsll(static_cast<__int64_t>(x)); }

/* Simply Runs 'ExtractMsbPos' and applies 'ConvertToReversedPos' on it */
FAST_DCALL_ALWAYS int ExtractLsbReversedPos(const __uint64_t x) { return ConvertToReversedPos(ExtractLsbPos(x)); }

/* Function does nothing, simply returns given value */
FAST_CALL constexpr int NoOp(const int m) { return m; }

/* Functions returns BitMap with MsbPos as 1 */
FAST_DCALL_ALWAYS __uint64_t ExtractMsbBitBuiltin(const __uint64_t x) { return cuda_MaxMsbPossible >> ExtractMsbPos(x); }

/* Functions returns BitMap with LsbPos as 1 */
FAST_DCALL_ALWAYS __uint64_t ExtractLsbBitBuiltin(const __uint64_t x) { return cuda_MinMsbPossible << ExtractLsbReversedPos(x); }

/* Functions returns BitMap with LsbPos as 1 */
FAST_CALL constexpr __uint64_t ExtractLsbOwn1(const __uint64_t x) { return x & -x; }

/* Functions returns BitMap with MsbPos as 1, with additional check whether given argument is 0 */
FAST_DCALL_ALWAYS __uint64_t ExtractMsbBit(const __uint64_t x) { return x == 0 ? 0 : ExtractMsbBitBuiltin(x); }

FAST_CALL constexpr __uint64_t ExtractMsbBitConstexpr(const __uint64_t x) {
    return x == 0 ? 0 : (cuda_MaxMsbPossible >> ExtractMsbPosConstexpr(x));
}

/* Functions returns BitMap with LsbPos as 1 */
FAST_CALL constexpr __uint64_t ExtractLsbBit(const __uint64_t x) { return ExtractLsbOwn1(x); }

/* Function simply returns a map with remove all bits that are present on b*/
FAST_CALL constexpr __uint64_t ClearAFromIntersectingBits(const __uint64_t a, const __uint64_t b) { return a ^ (a & b); }

/* Function simply counts ones on the given BitMap*/
FAST_DCALL_ALWAYS int CountOnesInBoard(const __uint64_t bitMap) { return __popcll(bitMap); }

/* Count same bits */
FAST_DCALL_ALWAYS int CountSameBits(const __uint64_t a, const __uint64_t b) { return __popcll((a & b) | ((~a) & (~b))); }

/*          IMPORTANT NOTES:
 *  Function assumes that containerPos is already set to 1
 *  and container[0] is 0 value, which will induce all others.
 *  Use GenerateBitPermutation() wrapper instead.
 */

template<class IndexableT>
HYBRID constexpr void
GenerateBitPermutationsRecursion(const __uint64_t number, const int bitPos, IndexableT &container, size_t &containerPos) {
    if (bitPos == -1 || number == 0)
        return;
    __uint64_t nextBit{};

    for (int i = bitPos; i >= 0; --i)
        if (const __uint64_t bitMap = 1LLU << i; (bitMap & number) != 0) {
            GenerateBitPermutationsRecursion(number ^ bitMap, i - 1, container, containerPos);
            nextBit = bitMap;
            break;
        }

    const size_t rangeEnd = containerPos;
    for (size_t i = 0; i < rangeEnd; ++i) {
        container[rangeEnd + i] = container[i] | nextBit;
    }

    containerPos *= 2;
}

template<class IndexableT>
FAST_CALL constexpr size_t GenerateBitPermutations(const __uint64_t number, IndexableT &container) {
    container[0] = 0;
    size_t index = 1;

    GenerateBitPermutationsRecursion(number, 63, container, index);
    return index;
}

#endif // CUDA_BIT_OPERATIONS_H
