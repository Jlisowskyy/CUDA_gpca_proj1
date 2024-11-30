//
// Created by Jlisowskyy on 3/7/24.
//

#ifndef STACK_H
#define STACK_H

#include <cstdlib>

#include "Helpers.cuh"

template<class ItemT>
struct Stack {
    // ------------------------------
    // Class inner types
    // ------------------------------

    struct StackPayload {
        ItemT *data;
        __uint32_t size;

        FAST_DCALL_ALWAYS void Push(Stack &s, ItemT item) {
            s.Push(item);
            ++size;
        }

        FAST_DCALL_ALWAYS const ItemT &operator[](__uint32_t ind) const { return data[ind]; }

        FAST_DCALL_ALWAYS ItemT &operator[](__uint32_t ind) { return data[ind]; }
    };

    // ------------------------------
    // Class creation
    // ------------------------------

    Stack() = delete;

    FAST_DCALL_ALWAYS explicit Stack(void *ptr, bool ClearStack = true) :
            _last(static_cast<__uint32_t *>(ptr)),
            _data(static_cast<ItemT *>(ptr) + 1) {
        if (ClearStack) {
            *_last = 0;
        }
    }

    ~Stack() = default;

    Stack(const Stack &) = delete;

    Stack(Stack &&) = delete;

    Stack &operator=(const Stack &) = delete;

    Stack &operator=(Stack &&) = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    FAST_DCALL_ALWAYS void Push(const ItemT item) {
        __uint32_t idx = atomicAdd(_last, 1);
        _data[idx] = item;
    }

    FAST_DCALL_ALWAYS void Clear() { *_last = 0; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint32_t Size() const { return *_last; }

    FAST_DCALL_ALWAYS const ItemT &operator[](__uint32_t ind) const { return _data[ind]; }

    FAST_DCALL_ALWAYS ItemT &operator[](__uint32_t ind) { return _data[ind]; }

    FAST_DCALL_ALWAYS ItemT *Top() { return _data + *_last; }

    // ------------------------------
    // Aggregates
    // ------------------------------

    /* Aggregates are not thread safe */

    FAST_DCALL_ALWAYS StackPayload GetPayload() { return {_data + *_last, 0}; }

    FAST_DCALL_ALWAYS void PopAggregate(const StackPayload payload) { *_last -= payload.size; }

    // ------------------------------
    // Class fields
    // ------------------------------

private:
    __uint32_t *_last;
    ItemT *_data;
};

#endif // STACK_H
