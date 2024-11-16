//
// Created by Jlisowskyy on 12/29/23.
//

#ifndef HASHMAP_H
#define HASHMAP_H

#include "Helpers.cuh"

__device__ static constexpr size_t MASKS_COUNT = 4;
__device__ static constexpr __uint64_t EMPTY_FIELD = ~0ULL;

template<size_t SIZE = 256>
class BaseMoveHashMap {

    // ------------------------------
    // Class creation
    // ------------------------------

public:

    constexpr BaseMoveHashMap() = default;

    constexpr ~BaseMoveHashMap() = default;

    __device__ constexpr explicit BaseMoveHashMap(const __uint64_t *nMasks, __uint64_t nMagic, __uint64_t nShift)
            : m_magic(nMagic), m_shift(nShift),
                m_fullMask(nMasks[0] | nMasks[1] | nMasks[2] | nMasks[3]), m_map{0} {
        for (int i = 0; i < MASKS_COUNT; i++) {
            m_masks[i] = nMasks[i];
        }
    }

    __device__ constexpr BaseMoveHashMap(const BaseMoveHashMap &other) {
        m_magic = other.m_magic;
        m_shift = other.m_shift;
        m_fullMask = other.m_fullMask;

        for (int i = 0; i < MASKS_COUNT; i++) {
            m_masks[i] = other.m_masks[i];
        }

        for (size_t i = 0; i < SIZE; ++i) {
            m_map[i] = other.m_map[i];
        }
    }

    __device__ constexpr BaseMoveHashMap(BaseMoveHashMap &&other)  noexcept {
        m_magic = other.m_magic;
        m_shift = other.m_shift;
        m_fullMask = other.m_fullMask;
        m_masks = other.m_masks;

        for (size_t i = 0; i < SIZE; ++i) {
            m_map[i] = other.m_map[i];
        }

        other.m_magic = 0;
        other.m_shift = 0;
        other.m_masks = nullptr;
        other.m_fullMask = 0;
        for (size_t i = 0; i < SIZE; ++i) {
            other.m_map[i] = 0;
        }
    }

    __device__ constexpr BaseMoveHashMap &operator=(const BaseMoveHashMap &other) {
        if (this != &other) {
            m_magic = other.m_magic;
            m_shift = other.m_shift;
            m_fullMask = other.m_fullMask;

            for (int i = 0; i < MASKS_COUNT; i++) {
                m_masks[i] = other.m_masks[i];
            }

            for (size_t i = 0; i < SIZE; ++i) {
                m_map[i] = other.m_map[i];
            }
        }
        return *this;
    }

    __device__ constexpr BaseMoveHashMap &operator=(BaseMoveHashMap &&other)  noexcept {
        if (this != &other) {
            m_magic = other.m_magic;
            m_shift = other.m_shift;
            m_fullMask = other.m_fullMask;
            m_masks = other.m_masks;

            for (size_t i = 0; i < SIZE; ++i) {
                m_map[i] = other.m_map[i];
            }

            other.m_magic = 0;
            other.m_shift = 0;
            other.m_masks = nullptr;
            other.m_fullMask = 0;
            for (size_t i = 0; i < SIZE; ++i) {
                other.m_map[i] = 0;
            }
        }
        return *this;
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] __device__ constexpr __uint64_t hashFunc(const __uint64_t val) const {
        return (val * m_magic) >> (64LLU - m_shift);
    }

    [[nodiscard]] __device__ constexpr const __uint64_t &operator[](const __uint64_t neighbors) const {
        return m_map[hashFunc(neighbors)];
    }

    [[nodiscard]] __device__ constexpr __uint64_t &operator[](const __uint64_t neighbors) {
        return m_map[hashFunc(neighbors)];
    }

    [[nodiscard]] __device__ constexpr __uint64_t getFullMask() const { return m_fullMask; }

    [[nodiscard]] __device__ constexpr const __uint64_t *getMasks() const { return m_masks; }

    // ------------------------------
    // Class fields
    // ------------------------------

protected:

    /* Hash function */
    __uint64_t m_magic{};
    __uint64_t m_shift{};

    /* Map components */
    __uint64_t *m_masks{};
    __uint64_t m_fullMask{};

    __uint64_t m_map[SIZE]{};
};

#endif // HASHMAP_H
