//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_CUDA_ARRAY_CUH
#define SRC_CUDA_ARRAY_CUH

#include <cuda_runtime.h>

#include <initializer_list>

#include "Helpers.cuh"

template<class T, size_t Size>
class cuda_Array final {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    constexpr cuda_Array() = default;

    constexpr ~cuda_Array() = default;

    HYBRID constexpr explicit cuda_Array(T *data) {
        for (size_t i = 0; i < Size; ++i) {
            m_data[i] = data[i];
        }
    }

    HYBRID constexpr cuda_Array(const cuda_Array &other) {
        for (size_t i = 0; i < Size; ++i) {
            m_data[i] = other.m_data[i];
        }
    }

    HYBRID constexpr cuda_Array(cuda_Array &&other) noexcept {
        for (size_t i = 0; i < Size; ++i) {
            m_data[i] = other.m_data[i];
        }
    }

    HYBRID constexpr cuda_Array &operator=(const cuda_Array &other) {
        if (this == &other) {
            return *this;
        }

        for (size_t i = 0; i < Size; ++i) {
            m_data[i] = other.m_data[i];
        }

        return *this;
    }

    HYBRID constexpr cuda_Array &operator=(cuda_Array &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        for (size_t i = 0; i < Size; ++i) {
            m_data[i] = other.m_data[i];
        }

        return *this;
    }

    HYBRID constexpr explicit cuda_Array(const T *data) {
        for (size_t i = 0; i < Size; ++i) {
            m_data[i] = data[i];
        }
    }

    HYBRID constexpr cuda_Array(std::initializer_list<T> init) {
        if (init.size() != Size) {
            return;
        }

        auto it = init.begin();
        for (size_t i = 0; i < Size; ++i, ++it) {
            m_data[i] = *it;
        }
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    __forceinline__ HYBRID constexpr T &operator[](const size_t index) { return m_data[index]; }

    __forceinline__ HYBRID constexpr const T &operator[](const size_t index) const { return m_data[index]; }

    __forceinline__ HYBRID constexpr T *data() { return m_data; }

    // ------------------------------
    // Class fields
    // ------------------------------
protected:
    T m_data[Size]{};
};

#endif //SRC_CUDA_ARRAY_CUH
