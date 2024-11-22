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
        size_t size;

        FAST_DCALL void Push(Stack &s, ItemT item) {
            s.Push(item);
            ++size;
        }

        FAST_DCALL const ItemT &operator[](size_t ind) const { return data[ind]; }

        FAST_DCALL ItemT &operator[](size_t ind) { return data[ind]; }
    };

    // ------------------------------
    // Class creation
    // ------------------------------

    Stack() = delete;

    FAST_DCALL explicit Stack(void *ptr) : _data(static_cast<ItemT *>(ptr)) {}

    ~Stack() = default;

    Stack(const Stack &) = delete;

    Stack(Stack &&) = delete;

    Stack &operator=(const Stack &) = delete;

    Stack &operator=(Stack &&) = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    FAST_DCALL void Push(const ItemT item) { _data[_last++] = item; }

    FAST_DCALL StackPayload GetPayload() { return {_data + _last, 0}; }

    FAST_DCALL void PopAggregate(const StackPayload payload) { _last -= payload.size; }

    FAST_DCALL ItemT Pop() { return _data[--_last]; }

    FAST_DCALL ItemT& Top() { return _data[_last - 1]; }

    FAST_DCALL void Clear() { _last = 0; }

    [[nodiscard]] FAST_DCALL size_t Size() const { return _last; }
    // ------------------------------
    // Class fields
    // ------------------------------

private:
    size_t _last{};
    ItemT *_data;
};

#endif // STACK_H
