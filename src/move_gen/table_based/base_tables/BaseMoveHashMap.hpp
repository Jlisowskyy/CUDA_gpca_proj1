//
// Created by Jlisowskyy on 12/29/23.
//

#ifndef HASHMAP_H
#define HASHMAP_H

#include <array>
#include <format>
#include <mutex>
#include <vector>

template<size_t SIZE = 256>
class BaseMoveHashMap {
    using _masksT = std::array<uint64_t, 4>;

    // ------------------------------
    // Class creation
    // ------------------------------

public:
    static constexpr size_t MASKS_COUNT = 4;

    constexpr BaseMoveHashMap() = default;

    constexpr ~BaseMoveHashMap() = default;

    constexpr explicit BaseMoveHashMap(const _masksT &nMasks, uint64_t nMagic,
                                       uint64_t nShift)
            : m_magic(nMagic), m_shift(nShift),  m_masks(nMasks),
                m_fullMask(nMasks[0] | nMasks[1] | nMasks[2] | nMasks[3]), m_map{0} {
    }

    constexpr BaseMoveHashMap(const BaseMoveHashMap &) = default;

    constexpr BaseMoveHashMap(BaseMoveHashMap &&) = default;

    constexpr BaseMoveHashMap &operator=(const BaseMoveHashMap &) = default;

    constexpr BaseMoveHashMap &operator=(BaseMoveHashMap &&) = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] constexpr uint64_t hashFunc(const uint64_t val) const { return (val * m_magic) >> (64LLU - m_shift); }

    [[nodiscard]] constexpr const uint64_t &operator[](const uint64_t neighbors) const {
        return m_map[hashFunc(neighbors)];
    }

    [[nodiscard]] constexpr uint64_t &operator[](const uint64_t neighbors) { return m_map[hashFunc(neighbors)]; }

    [[nodiscard]] constexpr uint64_t getFullMask() const { return m_fullMask; }

    [[nodiscard]] constexpr const _masksT& getMasks() const { return m_masks; }

    // ------------------------------
    // Class fields
    // ------------------------------

protected:
    static constexpr uint64_t EMPTY_FIELD = ~0ULL;

    /* Hash function */
    uint64_t m_magic{};
    uint64_t m_shift{};

    /* Map components */
    _masksT m_masks{};
    uint64_t m_fullMask{};

    std::array<uint64_t, SIZE> m_map{};
};

#endif // HASHMAP_H
