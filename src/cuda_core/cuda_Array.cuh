//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_CUDA_ARRAY_CUH
#define SRC_CUDA_ARRAY_CUH

#include <cuda_runtime.h>

#include <initializer_list>

#include "Helpers.cuh"

template<class T, __uint32_t SIZE>
class cuda_Array final {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    constexpr cuda_Array() = default;

    constexpr ~cuda_Array() = default;

    HYBRID constexpr explicit cuda_Array(T *data) {
        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = data[i];
        }
    }

    HYBRID constexpr cuda_Array(const cuda_Array &other) {
        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = other.m_data[i];
        }
    }

    HYBRID constexpr cuda_Array(cuda_Array &&other) noexcept {
        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = other.m_data[i];
        }
    }

    HYBRID constexpr cuda_Array &operator=(const cuda_Array &other) {
        if (this == &other) {
            return *this;
        }

        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = other.m_data[i];
        }

        return *this;
    }

    HYBRID constexpr cuda_Array &operator=(cuda_Array &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = other.m_data[i];
        }

        return *this;
    }

    HYBRID constexpr explicit cuda_Array(const T *data) {
        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = data[i];
        }
    }

    HYBRID constexpr cuda_Array(std::initializer_list<T> init) {
        if (init.size() != SIZE) {
            return;
        }

        auto it = init.begin();
        for (__uint32_t i = 0; i < SIZE; ++i, ++it) {
            m_data[i] = *it;
        }
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    __forceinline__ HYBRID constexpr T &operator[](const __uint32_t index) {
        assert(index < SIZE && "OVERFLOW!");
        return m_data[index];
    }

    __forceinline__ HYBRID constexpr const T &operator[](const __uint32_t index) const {
        assert(index < SIZE && "OVERFLOW!");
        return m_data[index];
    }

    __forceinline__ HYBRID constexpr T *data() { return m_data; }

    // ------------------------------
    // Class fields
    // ------------------------------
protected:
    T m_data[SIZE]{};
};

#endif //SRC_CUDA_ARRAY_CUH
