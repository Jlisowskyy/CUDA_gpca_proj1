#ifndef CUDA_BIT_OPERATIONS_H
#define CUDA_BIT_OPERATIONS_H

#include "Helpers.cuh"

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

/* This values should be used inside every shifting operation to ensure correct types under the operation */
__device__ __constant__ static constexpr uint64_t cuda_MinMsbPossible = static_cast<uint64_t>(1);
__device__ __constant__ static constexpr uint64_t cuda_MaxMsbPossible = cuda_MinMsbPossible << 63;

/* Function efficiently computes MsbPos */
FAST_DCALL_ALWAYS int ExtractMsbPos(const uint64_t x) { return __clzll(static_cast<int64_t>(x)); }

FAST_CALL_ALWAYS int ExtractMsbPosNeutral(const uint64_t x) {
    if (x == 0) return 63;

    uint64_t val = x;
    int pos = 0;

    /* Binary search approach to find MSB position */
    if (val > 0xFFFFFFFF) {
        val >>= 32;
        pos += 32;
    }
    if (val > 0xFFFF) {
        val >>= 16;
        pos += 16;
    }
    if (val > 0xFF) {
        val >>= 8;
        pos += 8;
    }
    if (val > 0xF) {
        val >>= 4;
        pos += 4;
    }
    if (val > 0x3) {
        val >>= 2;
        pos += 2;
    }
    if (val > 0x1) {
        val >>= 1;
        pos += 1;
    }

    return 63 - pos;
}

HYBRID constexpr int ExtractMsbPosConstexpr(const uint64_t x) {
    if (x == 0) {
        return 0;
    }

    uint64_t cursor = cuda_MaxMsbPossible;
    int count = 0;

    while ((cursor & x) == 0) {
        cursor >>= 1;
        ++count;
    }

    return count;
}

/* Nice way to reverse from MsbPos to LsbPos and other way around */
FAST_CALL_ALWAYS constexpr uint32_t ConvertToReversedPos(const uint32_t x) {
    return x ^ 63; // equals to 63 - x;
}

template<typename NumT>
FAST_CALL_ALWAYS constexpr void SetBitBoardBit(NumT &bitBoard, const uint32_t pos, const bool value) {
    bitBoard = (value << pos) | (bitBoard & ~(cuda_MinMsbPossible << pos));
}

FAST_CALL_ALWAYS constexpr uint32_t SwapColor(const uint32_t col) { return col ^ 1; }

/* Simply Runs 'ExtractMsbPos' and applies 'ConvertToReversedPos' on it */
FAST_DCALL_ALWAYS int ExtractMsbReversedPos(const uint64_t x) { return ConvertToReversedPos(ExtractMsbPos(x)); }

/* Function efficiently computes LsbPos */
FAST_DCALL_ALWAYS int ExtractLsbPos(const uint64_t x) { return __ffsll(static_cast<int64_t>(x)); }

/* Simply Runs 'ExtractMsbPos' and applies 'ConvertToReversedPos' on it */
FAST_DCALL_ALWAYS int ExtractLsbReversedPos(const uint64_t x) { return ConvertToReversedPos(ExtractLsbPos(x)); }

/* Function does nothing, simply returns given value */
FAST_CALL_ALWAYS constexpr int NoOp(const int m) { return m; }

/* Functions returns BitMap with MsbPos as 1 */
FAST_DCALL_ALWAYS uint64_t ExtractMsbBitBuiltin(const uint64_t x) {
    return cuda_MaxMsbPossible >> ExtractMsbPos(x);
}

/* Functions returns BitMap with LsbPos as 1 */
FAST_DCALL_ALWAYS uint64_t ExtractLsbBitBuiltin(const uint64_t x) {
    return cuda_MinMsbPossible << ExtractLsbReversedPos(x);
}

/* Functions returns BitMap with LsbPos as 1 */
FAST_CALL_ALWAYS constexpr uint64_t ExtractLsbOwn1(const uint64_t x) { return x & -x; }

/* Functions returns BitMap with MsbPos as 1, with additional check whether given argument is 0 */
FAST_DCALL_ALWAYS uint64_t ExtractMsbBit(const uint64_t x) { return x == 0 ? 0 : ExtractMsbBitBuiltin(x); }

FAST_CALL_ALWAYS constexpr uint64_t ExtractMsbBitConstexpr(const uint64_t x) {
    return x == 0 ? 0 : (cuda_MaxMsbPossible >> ExtractMsbPosConstexpr(x));
}

/* Functions returns BitMap with LsbPos as 1 */
FAST_CALL_ALWAYS constexpr uint64_t ExtractLsbBit(const uint64_t x) { return ExtractLsbOwn1(x); }

/* Function simply returns a map with remove all bits that are present on b*/
FAST_CALL_ALWAYS constexpr uint64_t ClearAFromIntersectingBits(const uint64_t a, const uint64_t b) {
    return a ^ (a & b);
}

/* Function simply counts ones on the given BitMap*/
FAST_DCALL_ALWAYS int CountOnesInBoard(const uint64_t bitMap) { return __popcll(bitMap); }

/* Count same bits */
FAST_DCALL_ALWAYS int CountSameBits(const uint64_t a, const uint64_t b) {
    return __popcll((a & b) | ((~a) & (~b)));
}

/*          IMPORTANT NOTES:
 *  Function assumes that containerPos is already set to 1
 *  and container[0] is 0 value, which will induce all others.
 *  Use GenerateBitPermutation() wrapper instead.
 */

template<class IndexableT>
HYBRID constexpr void
GenerateBitPermutationsRecursion(const uint64_t number, const int bitPos, IndexableT &container,
                                 uint32_t &containerPos) {
    if (bitPos == -1 || number == 0) {
        return;
    }
    uint64_t nextBit{};

    for (int i = bitPos; i >= 0; --i) {
        if (const uint64_t bitMap = 1LLU << i; (bitMap & number) != 0) {
            GenerateBitPermutationsRecursion(number ^ bitMap, i - 1, container, containerPos);
            nextBit = bitMap;
            break;
        }
    }

    const uint32_t rangeEnd = containerPos;
    for (uint32_t i = 0; i < rangeEnd; ++i) {
        container[rangeEnd + i] = container[i] | nextBit;
    }

    containerPos *= 2;
}

template<class IndexableT>
FAST_CALL_ALWAYS constexpr uint32_t GenerateBitPermutations(const uint64_t number, IndexableT &container) {
    container[0] = 0;
    uint32_t index = 1;

    GenerateBitPermutationsRecursion(number, 63, container, index);
    return index;
}

#endif // CUDA_BIT_OPERATIONS_H
