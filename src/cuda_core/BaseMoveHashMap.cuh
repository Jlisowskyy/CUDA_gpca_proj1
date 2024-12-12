#ifndef HASHMAP_H
#define HASHMAP_H

#include "cuda_Array.cuh"
#include "Helpers.cuh"

__device__ __constant__ static constexpr uint32_t MASKS_COUNT = 4;

template<uint32_t SIZE = 256>
class BaseMoveHashMap final {

    // ------------------------------
    // Class creation
    // ------------------------------

public:

    constexpr BaseMoveHashMap() = default;

    constexpr ~BaseMoveHashMap() = default;

    HYBRID constexpr explicit BaseMoveHashMap(const cuda_Array<uint64_t, MASKS_COUNT> &nMasks, const uint64_t nMagic,
                                              const uint64_t nShift)
            : m_magic(nMagic), m_shift(nShift), m_fullMask(nMasks[0] | nMasks[1] | nMasks[2] | nMasks[3]),
              m_masks(nMasks), m_map{0} {
    }

    constexpr BaseMoveHashMap(const BaseMoveHashMap &) = default;

    constexpr BaseMoveHashMap(BaseMoveHashMap &&) = default;

    constexpr BaseMoveHashMap &operator=(const BaseMoveHashMap &) = default;

    constexpr BaseMoveHashMap &operator=(BaseMoveHashMap &&) = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] HYBRID constexpr uint64_t hashFunc(const uint64_t val) const {
        return (val * m_magic) >> (64LLU - m_shift);
    }

    [[nodiscard]] HYBRID constexpr const uint64_t &operator[](const uint64_t neighbors) const {
        return m_map[hashFunc(neighbors)];
    }

    [[nodiscard]] HYBRID constexpr uint64_t &operator[](const uint64_t neighbors) {
        return m_map[hashFunc(neighbors)];
    }

    [[nodiscard]] HYBRID constexpr uint64_t getFullMask() const { return m_fullMask; }

    [[nodiscard]] HYBRID constexpr const cuda_Array<uint64_t, MASKS_COUNT> &getMasks() const { return m_masks; }

    // ------------------------------
    // Class fields
    // ------------------------------

protected:

    /* Hash function */
    uint64_t m_magic{};
    uint64_t m_shift{};
    uint64_t m_fullMask{};

    /* Map components */
    alignas(128) cuda_Array<uint64_t, MASKS_COUNT> m_masks{};

    alignas(128) cuda_Array<uint64_t, SIZE> m_map{};
};

#endif // HASHMAP_H
