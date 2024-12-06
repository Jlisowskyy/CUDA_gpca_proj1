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

#include <cuda_runtime.h>

#include <random>
#include <thread>
#include <iostream>

/**
 * @brief Monte Carlo Tree Search (MCTS) Engine template class for game AI
 *
 * This class implements a flexible Monte Carlo Tree Search algorithm that can
 * operate on both CPU and GPU, supporting configurable batch sizes and worker threads.
 *
 * @tparam BATCH_SIZE Number of simulations to play in single batch
 * @tparam ENGINE_TYPE Specifies whether to use CPU or GPU expansion strategy
 */
template<EngineType ENGINE_TYPE = EngineType::GPU1, bool USE_TIMERS = false>
class MctsEngine {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    MctsEngine() = delete;

    MctsEngine(MctsEngine &) = delete;

    MctsEngine(MctsEngine &&) = delete;

    /**
     * @brief Construct a new MctsEngine with a given board state
     *
     * @param board Initial game board state
     * @param numWorkers Number of worker threads for tree expansion (default: 1)
     */
    explicit MctsEngine(const cuda_Board &board, const uint32_t numWorkers = 1) : m_board(board),
                                                                                    m_numWorkers(numWorkers),
                                                                                    m_pool(numWorkers) {
        cpu::AllocateStacks(numWorkers);
    }

    ~MctsEngine() {
        delete m_root;
        cpu::DeallocStacks(m_numWorkers);
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    /**
     * @brief Initiates the move search process
     *
     * Initializes the search tree, starts worker threads for tree expansion,
     * and begins the Monte Carlo Tree Search algorithm
     */
    void MoveSearchStart() {
        _initTree();

        /* Reset performance counters */
        mcts::g_ExpandRacesCounter.store(0);
        mcts::g_SimulationCounter.store(0);
        mcts::g_CopyBackTimes.store(0.0);
        mcts::g_CopyTimes.store(0.0);
        mcts::g_KernelTime.store(0.0);

        m_shouldWork = true;
        m_pool.Reset(m_numWorkers);
        m_pool.RunThreads(_worker, this);
    }

    /**
     * @brief Waits for move search to complete and selects the best move
     *
     * @return cuda_Move The best move found during the search
     */
    [[nodiscard]] cuda_Move MoveSearchWait() {
        m_shouldWork = false;
        m_pool.Wait();

        return _pickMove();
    }

    /**
     * @brief Applies a move to the current board state and adapts the search tree
     *
     * Updates the root of the search tree to reflect the new board state
     * after a move has been made
     *
     * @param move The move to apply
     */
    void ApplyMove(const cuda_Move move) {
        assert(move.IsOkayMoveCPU() && "ENGINE RECEIVED MALFUNCTIONING MOVE!");

        cuda_Move::MakeMove(move, m_board);
        _adaptTree(move);
    }

    [[nodiscard]] cuda_Move GetCurrentBestMove() const {
        return _pickMove();
    }

    void DisplayResults() {
        const auto pickedMove = GetCurrentBestMove();

        std::cout << "Engine picked move: " << pickedMove.GetPackedMove().GetLongAlgebraicNotation()
                  << " with total: " << mcts::g_SimulationCounter.load()
                  << " simulations made and total expansion races: " << mcts::g_ExpandRacesCounter.load()
                  << std::endl;

        std::cout << "Performance metrics (average):" << std::endl
                << "  Copy times:     " << 1000.0 * mcts::g_CopyTimes.load() / double(mcts::g_SimulationCounter.load())
                << " milliseconds" << std::endl
                << "  Copy back times: "
                << 1000.0 * mcts::g_CopyBackTimes.load() / double(mcts::g_SimulationCounter.load())
                << " milliseconds" << std::endl
                << "  Kernel times:   " << 1000.0 * mcts::g_KernelTime.load() / double(mcts::g_SimulationCounter.load())
                << " milliseconds" << std::endl;
    }

    [[nodiscard]] static constexpr const char *GetName() {
        switch (ENGINE_TYPE) {
            case EngineType::GPU1:
                return "GPU0";
            case EngineType::GPU0:
                return "GPU1";
            case EngineType::CPU:
                return "CPU";
            default:
                std::abort();
        }
    }

    [[nodiscard]] static uint32_t GetPreferredThreadsCount() {
        switch (ENGINE_TYPE) {
            case EngineType::GPU1:
                return std::thread::hardware_concurrency() * 6;
            case EngineType::GPU0:
                return std::thread::hardware_concurrency() * 6;
            case EngineType::CPU:
                return 1;
            default:
                std::abort();
        }
    }

    // ------------------------------
    // Class implementation
    // ------------------------------
protected:

    /**
     * @brief Worker thread function for tree expansion
     *
     * Expands the search tree using either GPU or CPU methods
     * based on the ENGINE_TYPE template parameter
     *
     * @param idx Worker thread index
     * @param workspace Pointer to the MctsEngine instance
     */
    static void _worker(uint32_t idx, MctsEngine *workspace) {
        cudaStream_t stream;

        if constexpr (ENGINE_TYPE == EngineType::GPU1 || ENGINE_TYPE == EngineType::GPU0) {
            CUDA_ASSERT_SUCCESS(cudaStreamCreate(&stream));
        }

        /* expand the tree until action is revoked */
        while (workspace->m_shouldWork) {
            if constexpr (ENGINE_TYPE == EngineType::GPU1 || ENGINE_TYPE == EngineType::GPU0) {
                mcts::ExpandTreeGPU<ENGINE_TYPE, USE_TIMERS>(workspace->m_root, stream);
            } else if constexpr (ENGINE_TYPE == EngineType::CPU) {
                mcts::ExpandTreeCPU(workspace->m_root);
            } else {
                std::abort();
            }
        }

        if constexpr (ENGINE_TYPE == EngineType::GPU1 || ENGINE_TYPE == EngineType::GPU0) {
            CUDA_ASSERT_SUCCESS(cudaStreamDestroy(stream));
        }

#ifdef WRITE_OUT
        std::cout << "worker with id " << idx << " died..." << std::endl;
#endif
    }

    /**
     * @brief Initializes the search tree with possible moves
     *
     * Generates all possible moves and adds them as child nodes
     * to the root of the search tree
     */
    void _initTree() {
        /* Prepare root if necessary */
        if (!m_root) {
            m_root = new MctsNode(m_board);
        }

        /* leave if children are already added */
        if (m_root->HasChildrenAssigned()) {
            return;
        }

        /* generate all possible children */
        const auto moves = ported_translation::GenMoves(m_board);

        /* Alloc the vector */
        const auto pChildren = new std::vector<MctsNode *>();
        pChildren->reserve(moves.size());

        /* Add child nodes */
        for (const auto move: moves) {
            pChildren->emplace_back(new MctsNode(m_root, move));
        }

        const bool result = m_root->SetChildren(pChildren);
        assert(result && "FAILED TO ADD CHILDREN AS INIT!");
    }

    /**
     * @brief Adapts the search tree after a move is applied
     *
     * Updates the root of the search tree to the corresponding
     * child node matching the applied move
     *
     * @param move The move that was applied
     */
    void _adaptTree(const cuda_Move move) {
        if (!m_root) {
            return;
        }

        MctsNode *newParent{};
        for (const auto child: m_root->GetChildren()) {
            if (child->m_move.GetPackedMove() == move.GetPackedMove()) {
                newParent = child;
                break;
            }
        }

        if (newParent) {
            newParent->m_parent = nullptr;

            for (auto &child: m_root->GetChildren()) {
                if (child != newParent) {
                    delete child;
                }

                child = nullptr;
            }
        }

        delete m_root;
        m_root = newParent;
    }

    /**
     * @brief Selects the best move based on calculated win rates
     *
     * Iterates through child nodes to find the move with the highest win rate
     *
     * @return cuda_Move The move with the highest calculated win rate
     */
    [[nodiscard]] cuda_Move _pickMove() const {
        assert(m_root != nullptr && "ENGINE: CALLED PICK MOVE ON EMPTY TREE!");

        cuda_Move bestMove{};
        double bestWinRate = std::numeric_limits<double>::min();

        for (const auto child: m_root->GetChildren()) {
            if (child->GetNumSamples() == 0) {
                continue;
            }

            if (const double winRate = child->CalculateWinRate(); winRate > bestWinRate) {
                bestWinRate = winRate;
                bestMove = child->m_move;
            }
        }

#ifdef WRITE_OUT
        std::cout << "PICKED MOVE: " << bestMove.GetPackedMove().GetLongAlgebraicNotation() << " with winrate: "
                  << bestWinRate << std::endl;
#endif

        return bestMove;
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    cuda_Board m_board{};
    MctsNode *m_root{};

    uint32_t m_numWorkers{};
    bool m_shouldWork{};
    ThreadPool m_pool;
};


#endif //SRC_MCTSENGINE_CUH
