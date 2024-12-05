//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_MCTSENGINE_CUH
#define SRC_MCTSENGINE_CUH

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"

#include "MctsTree.cuh"
#include "cpu_MoveGen.cuh"
#include "ThreadPool.cuh"

#include <random>
#include <thread>
#include <iostream>

static constexpr __uint32_t DEFAULT_MCTS_BATCH_SIZE = 64;

template<__uint32_t BATCH_SIZE>
class MctsEngine {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    explicit MctsEngine(const cuda_Board &board, const __uint32_t numWorkers = 1) : m_board(board),
                                                                                    m_numWorkers(numWorkers),
                                                                                    m_pool(numWorkers) {}

    // ------------------------------
    // Class interaction
    // ------------------------------

    void MoveSearchStart() {
        _initTree();

        m_shouldWork = true;
        m_pool.Reset(m_numWorkers);
        m_pool.RunThreads(_worker, this);
    }

    [[nodiscard]] cuda_Move MoveSearchWait() {
        m_shouldWork = false;
        m_pool.Wait();

        return _pickMove();
    }

    void ApplyMove(const cuda_Move move) {
        assert(move.IsOkayMoveCPU() && "ENGINE RECEIVED MALFUNCTIONING MOVE!");

        cuda_Move::MakeMove(move, m_board);
        _adaptTree(move);
    }

    // ------------------------------
    // Class implementation
    // ------------------------------
protected:

    static void _worker(__uint32_t idx, MctsEngine<BATCH_SIZE> *workspace) {
        while (workspace->m_shouldWork) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        std::cout << "worker with id " << idx << " died..." << std::endl;
    }

    void _initTree() {

    }

    void _adaptTree(const cuda_Move move) {

    }

    [[nodiscard]] cuda_Move _pickMove() {
        auto moves = ported_translation::GenMoves(m_board);
        std::shuffle(moves.begin(), moves.end(), std::mt19937_64(std::random_device()()));

        return moves[0];
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    __uint32_t m_moveTime{};
    cuda_Board m_board{};

    MctsNode *m_root{};

    __uint32_t m_numWorkers{};
    bool m_shouldWork{};
    ThreadPool m_pool;
};


#endif //SRC_MCTSENGINE_CUH
