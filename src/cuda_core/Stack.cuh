#ifndef STACK_H
#define STACK_H

#include <cstdlib>
#include <cstdint>

#include "Helpers.cuh"

template<class ItemT>
struct Stack {
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

    FAST_DCALL_ALWAYS explicit Stack(ItemT *ptr, __uint32_t *ctrPtr, bool ClearStack = true) :
            _last(ctrPtr), _data(ptr) {
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

    template<__uint32_t MAX_ITEMS = UINT32_MAX>
    FAST_DCALL_ALWAYS bool Push(const ItemT item) {
        __uint32_t idx = atomicAdd_block(_last, 1);

        if (MAX_ITEMS != UINT32_MAX && idx >= MAX_ITEMS) {
            atomicSub_block(_last, 1);
            return false;
        }

        _data[idx] = item;
        return true;
    }

    FAST_DCALL_ALWAYS void Clear() { *_last = 0; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint32_t Size() const { return *_last; }

    FAST_DCALL_ALWAYS const ItemT &operator[](__uint32_t ind) const {
        assert(ind < *_last && "STACK OVERFLOW!");
        return _data[ind];
    }

    FAST_DCALL_ALWAYS ItemT &operator[](__uint32_t ind) {
        assert(ind < *_last && "STACK OVERFLOW!");
        return _data[ind];
    }

    FAST_DCALL_ALWAYS ItemT *Top() { return _data + *_last; }

    // ------------------------------
    // Class fields
    // ------------------------------

private:
    __uint32_t *_last;
    ItemT *_data;
};

#endif // STACK_H
