//
// Created by Jlisowskyy on 12/29/23.
//

#ifndef HASHMAP_H
#define HASHMAP_H

#include "cuda_Array.cuh"
#include "Helpers.cuh"

__device__ static constexpr size_t MASKS_COUNT = 4;
__device__ static constexpr __uint64_t EMPTY_FIELD = ~0ULL;

template<size_t SIZE = 256>
class BaseMoveHashMap final {

    // ------------------------------
    // Class creation
    // ------------------------------

public:

    constexpr BaseMoveHashMap() = default;

    constexpr ~BaseMoveHashMap() = default;

    HYBRID constexpr explicit BaseMoveHashMap(const cuda_Array<__uint64_t, MASKS_COUNT> &nMasks, __uint64_t nMagic,
                                                  __uint64_t nShift)
            : m_masks(nMasks), m_magic(nMagic), m_shift(nShift),
                m_fullMask(nMasks[0] | nMasks[1] | nMasks[2] | nMasks[3]), m_map{0} {
    }

    constexpr BaseMoveHashMap(const BaseMoveHashMap &) = default;

    constexpr BaseMoveHashMap(BaseMoveHashMap &&) = default;

    constexpr BaseMoveHashMap &operator=(const BaseMoveHashMap &) = default;

    constexpr BaseMoveHashMap &operator=(BaseMoveHashMap &&) = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] HYBRID constexpr __uint64_t hashFunc(const __uint64_t val) const {
        return (val * m_magic) >> (64LLU - m_shift);
    }

    [[nodiscard]] HYBRID constexpr const __uint64_t &operator[](const __uint64_t neighbors) const {
        return m_map[hashFunc(neighbors)];
    }

    [[nodiscard]] HYBRID constexpr __uint64_t &operator[](const __uint64_t neighbors) {
        return m_map[hashFunc(neighbors)];
    }

    [[nodiscard]] HYBRID constexpr __uint64_t getFullMask() const { return m_fullMask; }

    [[nodiscard]] HYBRID constexpr const cuda_Array<__uint64_t, MASKS_COUNT> &getMasks() const { return m_masks; }

    // ------------------------------
    // Class fields
    // ------------------------------

protected:

    /* Hash function */
    __uint64_t m_magic{};
    __uint64_t m_shift{};

    /* Map components */
    cuda_Array<__uint64_t, MASKS_COUNT> m_masks{};
    __uint64_t m_fullMask{};

    cuda_Array<__uint64_t, SIZE> m_map{};
};

#endif // HASHMAP_H
