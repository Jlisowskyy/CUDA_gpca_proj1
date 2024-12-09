//
// Created by Jlisowskyy on 04/12/24.
//

#include "CpuMoveGen.h"

#include "engine/Checkmate-Chariot/include/MoveGeneration/BishopMap.h"
#include "engine/Checkmate-Chariot/include/MoveGeneration/RookMap.h"
#include "engine/Checkmate-Chariot/include/MoveGeneration/MoveGenerator.h"
#include "engine/Checkmate-Chariot/include/BitOperations.h"

#include <random>
#include <atomic>
#include <array>
#include <mutex>

/**
 * @brief Simple pseudo-random number generator with XOR-shift algorithm.
 *
 * Modifies the input state using bitwise XOR operations to generate the next random value.
 *
 * @param state Reference to the random state, which is modified in-place
 */
static void simpleRand(uint32_t &state) {
    static constexpr uint32_t COEF1 = 36969;
    static constexpr uint32_t COEF2 = 65535;
    static constexpr uint32_t COEF3 = 17;
    static constexpr uint32_t COEF4 = 13;

    state ^= state << 13;
    state *= COEF1;
    state += COEF2;
    state ^= state >> 7;
    state *= COEF3;
    state += COEF4;
    state ^= state << 17;
}


//template<typename T>
//class ThreadSafeStack {
//private:
//    struct Node {
//        T data;
//        std::atomic<Node *> next;
//
//        explicit Node(const T &value) : data(value), next(nullptr) {}
//    };
//
//    std::atomic<Node *> head{nullptr};
//public:
//    ~ThreadSafeStack() {
//        while (Node *oldHead = head.load()) {
//            head.store(oldHead->next);
//            delete oldHead;
//        }
//    }
//
//    void push(const T &value) {
//        Node *newNode = new Node(value);
//        Node *currentHead = head.load();
//
//        do {
//            newNode->next.store(head.load());
//        } while (!head.compare_exchange_weak(currentHead, newNode));
//    }
//
//    bool pop(T &result) {
//        Node *oldHead = head.load();
//
//        do {
//            if (oldHead == nullptr) {
//                return false;
//            }
//        } while (!head.compare_exchange_weak(oldHead, oldHead->next.load()));
//
//        result = oldHead->data;
//        delete oldHead;
//        return true;
//    }
//};

template<typename T, size_t STACK_SIZE>
class ThreadSafeStack {
private:
    std::array<T, STACK_SIZE> data{};
    size_t top{};

    std::mutex guard{};
public:

    void push(const T &value) {
        std::lock_guard<std::mutex> lock(guard);
        data[top++] = value;
    }

    bool pop(T &result) {
        std::lock_guard<std::mutex> lock(guard);
        result = data[--top];
        return true;
    }
};

using st = Stack<Move, DEFAULT_STACK_SIZE>;

ThreadSafeStack<st *, 1024> GlobalStacks{};

namespace cpu {
    uint64_t AccessCpuRookMap(const int msbInd, const uint64_t fullMap) {
        return RookMap::GetMoves(msbInd, fullMap);
    }

    uint64_t AccessCpuBishopMap(const int msbInd, const uint64_t fullMap) {
        return BishopMap::GetMoves(msbInd, fullMap);
    }

    int ExtractMsbPosCPU(const uint64_t map) {
        return ExtractMsbPos(map);
    }

    std::vector<external_move> GenerateMoves(const external_board &board) {
        const Board bd = TranslateToInternalBoard(board);

        st *s;
        const bool stackResult = GlobalStacks.pop(s);
        assert(stackResult);

        MoveGenerator mech{bd, *s};
        auto moves = mech.GetMovesFast();

        std::vector<external_move> result;
        result.reserve(moves.size);
        for (size_t i = 0; i < moves.size; ++i) {
            const auto &mv = moves[i];
            result.push_back({mv.GetPackedMove().DumpContent(), mv.DumpIndexes(), mv.DumpMisc()});
        }

        s->PopAggregate(moves);
        GlobalStacks.push(s);

        return result;
    }

    bool IsCheck(const external_board &board) {
        const Board bd = TranslateToInternalBoard(board);
        ChessMechanics mech{bd};
        return mech.IsCheck();
    }

    Board TranslateToInternalBoard(const external_board &exBoard) {
        Board bd{};
        for (int i = 0; i < 12; ++i) {
            bd.BitBoards[i] = exBoard[i];
        }
        bd.ElPassantField = exBoard[12];
        bd.Castlings = exBoard[13];
        bd.MovingColor = static_cast<int>(exBoard[14]);

        return bd;
    }

    static constexpr int32_t FIG_VALUES_CPU[Board::BitBoardsCount + 1]{
            100, 330, 330, 500, 900, 10000, -100, -330, -330, -500, -900, -10000, 0
    };

    uint32_t SimulateGame(const external_board &board) {
        static constexpr uint32_t MAX_DEPTH = 100;
        static constexpr uint32_t DRAW = 2;
        static constexpr uint32_t NUM_ROUNDS_IN_MATERIAL_ADVANTAGE_TO_WIN = 5;
        static constexpr uint32_t MATERIAL_ADVANTAGE_TO_WIN = 500;

        uint32_t evalCounters[2]{};
        uint32_t seed = std::mt19937{
                static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())}();

        st *s;
        const bool stackResult = GlobalStacks.pop(s);
        assert(stackResult);

        Board bd = TranslateToInternalBoard(board);
        MoveGenerator mech{bd, *s};

        for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth) {
            auto moves = mech.GetMovesFast();

            if (moves.size == 0) {
                GlobalStacks.push(s);
                return mech.IsCheck() ? SwapColor(bd.MovingColor) : DRAW;
            }

            const auto nextMove = moves[seed % moves.size];
            s->PopAggregate(moves);
            Move::MakeMove(nextMove, bd);

            uint32_t eval{};

            for (uint32_t bIdx = 0; bIdx < Board::BitBoardsCount; ++bIdx) {
                eval += CountOnesInBoard(bd.BitBoards[bIdx]) * FIG_VALUES_CPU[bIdx];
            }

            const uint32_t correctedEval = bd.MovingColor == BLACK ? -eval : eval;
            const bool isInWinningRange = correctedEval >= MATERIAL_ADVANTAGE_TO_WIN;
            evalCounters[bd.MovingColor] = isInWinningRange ? evalCounters[bd.MovingColor] + 1 : 0;

            if (evalCounters[bd.MovingColor] >= NUM_ROUNDS_IN_MATERIAL_ADVANTAGE_TO_WIN) {
                s->PopAggregate(moves);

                GlobalStacks.push(s);
                return bd.MovingColor;
            }


            simpleRand(seed);
        }

        GlobalStacks.push(s);
        return DRAW;
    }

    void AllocateStacks(const uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            GlobalStacks.push(new st{});
        }
    }

    void DeallocStacks(const uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            st *ptr;
            GlobalStacks.pop(ptr);
            delete ptr;
        }
    }
}
