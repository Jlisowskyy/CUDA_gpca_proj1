//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_CUDA_ARRAY_CUH
#define SRC_CUDA_ARRAY_CUH

#include <cuda_runtime.h>

#include <initializer_list>

#include "Helpers.cuh"

/**
 * @class cuda_Array
 * @brief A fixed-size array class with CUDA hybrid support for both host and device environments
 *
 * @tparam T The type of elements stored in the array
 * @tparam SIZE The compile-time fixed size of the array
 *
 * This class provides a lightweight, compile-time sized array that can be used
 * in both CUDA host and device code. It supports various construction methods,
 * including default, copy, move, and initializer list constructors.
 *
 * Key features:
 * - Compile-time fixed size
 * - HYBRID macro for host and device code compatibility
 * - Bounds-checked element access
 * - Support for initializer list construction
 * - constexpr support
 */
template<class T, __uint32_t SIZE>
class cuda_Array final {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    /**
     * @brief Default constructor
     *
     * Initializes the array with default-constructed elements
     */
    constexpr cuda_Array() = default;

    /**
     * @brief Destructor
     *
     * Default destructor that cleans up the array
     */
    constexpr ~cuda_Array() = default;

    /**
     * @brief Construct the array from a raw pointer
     *
     * @param data Pointer to the source data to copy from
     *
     * Copies SIZE elements from the provided pointer into the array
     */
    HYBRID constexpr explicit cuda_Array(T *data) {
        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = data[i];
        }
    }

    /**
     * @brief Copy constructor
     *
     * @param other The source array to copy from
     *
     * Creates a deep copy of the input array
     */
    HYBRID constexpr cuda_Array(const cuda_Array &other) {
        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = other.m_data[i];
        }
    }

    /**
     * @brief Move constructor
     *
     * @param other The source array to move from
     *
     * Moves elements from the source array to this array
     */
    HYBRID constexpr cuda_Array(cuda_Array &&other) noexcept {
        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = other.m_data[i];
        }
    }

    /**
     * @brief Copy assignment operator
     *
     * @param other The source array to copy from
     * @return Reference to the current array
     *
     * Performs a deep copy of the input array, avoiding self-assignment
     */
    HYBRID constexpr cuda_Array &operator=(const cuda_Array &other) {
        if (this == &other) {
            return *this;
        }

        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = other.m_data[i];
        }

        return *this;
    }

    /**
     * @brief Move assignment operator
     *
     * @param other The source array to move from
     * @return Reference to the current array
     *
     * Moves elements from the source array, avoiding self-assignment
     */
    HYBRID constexpr cuda_Array &operator=(cuda_Array &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = other.m_data[i];
        }

        return *this;
    }

    /**
     * @brief Construct the array from a constant raw pointer
     *
     * @param data Constant pointer to the source data to copy from
     *
     * Copies SIZE elements from the provided constant pointer into the array
     */
    HYBRID constexpr explicit cuda_Array(const T *data) {
        for (__uint32_t i = 0; i < SIZE; ++i) {
            m_data[i] = data[i];
        }
    }

    /**
     * @brief Construct the array from an initializer list
     *
     * @param init Initializer list containing elements to populate the array
     *
     * Creates the array from the given initializer list.
     * Requires the initializer list to have exactly SIZE elements.
     */
    HYBRID constexpr cuda_Array(std::initializer_list<T> init) {
        /* MAY BE BUG PRONE - BE CAUTIOUS HERE */
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

    /**
     * @brief Mutable element access operator
     *
     * @param index The index of the element to access
     * @return Reference to the element at the specified index
     *
     * Provides bounds-checked mutable access to array elements
     * Triggers an assertion if the index is out of bounds
     */
    FAST_CALL_ALWAYS constexpr T &operator[](const __uint32_t index) {
        assert(index < SIZE && "OVERFLOW!");
        return m_data[index];
    }

    /**
     * @brief Constant element access operator
     *
     * @param index The index of the element to access
     * @return Const reference to the element at the specified index
     *
     * Provides bounds-checked constant access to array elements
     * Triggers an assertion if the index is out of bounds
     */
    FAST_CALL_ALWAYS constexpr const T &operator[](const __uint32_t index) const {
        assert(index < SIZE && "OVERFLOW!");
        return m_data[index];
    }

    /**
     * @brief Get a pointer to the underlying data
     *
     * @return Pointer to the first element of the array
     *
     * Provides direct access to the array's internal storage
     */
    FAST_CALL_ALWAYS constexpr T *data() { return m_data; }

    // ------------------------------
    // Class fields
    // ------------------------------

protected:
    T m_data[SIZE]{};
};

#endif //SRC_CUDA_ARRAY_CUH
