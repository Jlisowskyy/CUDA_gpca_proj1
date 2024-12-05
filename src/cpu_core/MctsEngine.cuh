//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_MCTSENGINE_CUH
#define SRC_MCTSENGINE_CUH

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"

#include "MctsNode.cuh"
#include "cpu_MoveGen.cuh"
#include "ThreadPool.cuh"
#include "Mcts.cuh"

#include <random>
#include <thread>
#include <iostream>

static constexpr __uint32_t DEFAULT_MCTS_BATCH_SIZE = 64;
static constexpr __uint32_t MIN_SAMPLES_TO_EXPAND = 512;

enum class EngineType {
    CPU,
    GPU,
};

template<__uint32_t BATCH_SIZE, EngineType ENGINE_TYPE = EngineType::GPU>
class MctsEngine {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    MctsEngine() = delete;

    MctsEngine(MctsEngine &) = delete;

    MctsEngine(MctsEngine &&) = delete;

    explicit MctsEngine(const cuda_Board &board, const __uint32_t numWorkers = 1) : m_board(board),
                                                                                    m_numWorkers(numWorkers),
                                                                                    m_pool(numWorkers) {}

    ~MctsEngine() {
        delete m_root;
    }

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

    /* Performs MCT expansion */
    static void _worker(__uint32_t idx, MctsEngine *workspace) {
        /* expand the tree until action is revoked */
        while (workspace->m_shouldWork) {
            if constexpr (ENGINE_TYPE == EngineType::GPU) {
                mcts::ExpandTreeGPU<BATCH_SIZE>(workspace->m_root);
            } else {
                mcts::ExpandTreeCPU(workspace->m_root);
            }
        }
#ifdef WRITE_OUT
        std::cout << "worker with id " << idx << " died..." << std::endl;
#endif
    }

    void _initTree() {
        /* Prepare root if necessary */
        if (!m_root) {
            m_root = new MctsNode(m_board);
        }

        /* generate all possible children */
        const auto moves = ported_translation::GenMoves(m_board);

        /* Add missing child nodes */
        for (const auto move: moves) {
            bool isAlreadyAdded{};

            /* verify if the child is missing */
            for (const auto child: m_root->m_children) {
                if (child->m_move.GetPackedMove() == move.GetPackedMove()) {
                    isAlreadyAdded = true;
                    break;
                }
            }

            if (!isAlreadyAdded) {
                m_root->AddChildren(move);
            }
        }
    }

    void _adaptTree(const cuda_Move move) {
        if (!m_root) {
            return;
        }

        MctsNode *newParent{};
        for (const auto child: m_root->m_children) {
            if (child->m_move.GetPackedMove() == move.GetPackedMove()) {
                newParent = child;
                break;
            }
        }

        m_root = newParent;
    }

    [[nodiscard]] cuda_Move _pickMove() {
        assert(m_root != nullptr && "ENGINE: CALLED PICK MOVE ON EMPTY TREE!");

        cuda_Move bestMove{};
        double bestWinRate = std::numeric_limits<double>::min();

        for (const auto child: m_root->m_children) {
            if (const double winRate = child->CalculateWinRate(); winRate > bestWinRate) {
                bestWinRate = winRate;
                bestMove = child->m_move;
            }
        }

#ifdef WRITE_OUT
        std::cout << "PICKED MOVE: " << bestMove.GetPackedMove().GetLongAlgebraicNotation() << " with winrate: "
                  << bestWinRate << std::endl;
#endif
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    cuda_Board m_board{};
    MctsNode *m_root{};

    __uint32_t m_numWorkers{};
    bool m_shouldWork{};
    ThreadPool m_pool;
};


#endif //SRC_MCTSENGINE_CUH
