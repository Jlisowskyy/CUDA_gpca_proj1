//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_MCTSNODE_CUH
#define SRC_MCTSNODE_CUH

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"

#include <vector>
#include <algorithm>
#include <atomic>
#include <numeric>

static constexpr double UCB_COEF = 2.0;

/**
 * @brief Represents a node in the Monte Carlo Tree Search (MCTS) algorithm
 *
 * Manages game state, move tracking, scoring, and tree structure for MCTS
 * All operations on score are performed atomically
 */
class MctsNode {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    MctsNode() = delete;

    MctsNode(MctsNode &) = delete;

    MctsNode(MctsNode &&) = delete;

    /**
     * @brief Construct a root node with initial board state, intended
     * to be only used for root node
     *
     * @param board Initial game board configuration
     */
    explicit MctsNode(const cuda_Board &board) : m_board(board) {
        static constexpr __uint32_t INIT_SIZE = 32;
    }

    /**
     * @brief Construct a child node from a parent and move
     *
     * @param parent Parent node in the search tree
     * @param move Move that led to this node's board state from the parent board
     */
    explicit MctsNode(MctsNode *parent, const cuda_Move move) : m_parent(parent), m_move(move) {
        assert(parent && "MCTS NODE RECEIVED NULL PARENT");
        assert(move.IsOkayMoveCPU() && "MCTS NODE RECEIVED MALFUNCTIONED MOVE");

        m_board = parent->m_board;

        cuda_Move::MakeMove(move, m_board);
    }

    ~MctsNode() {
        Cleanup();
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    /**
     * @brief return number of samples done / being operated on atomically
     *
     * @return number of samples simulated or being worked on
     */
    [[nodiscard]] __uint32_t GetNumSamples() const {
        return m_numSamples.load();
    }

    /**
     * @brief Increment number of simulation samples atomically
     */
    void IncNumSamples() {
        m_numSamples.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard("DATA LEAK POSSIBLE")]] bool SetChildren(std::vector<MctsNode *> *children) {
        assert(children != nullptr && "MCTS NODE: DETECTEC NULLPTR CHILDREN");

        std::vector<MctsNode *> *expected = nullptr;
        return m_children.compare_exchange_strong(expected, children);
    }

    [[nodiscard]] bool HasChildrenAssigned() const {
        return m_children.load() != nullptr;
    }

    [[nodiscard]] const std::vector<MctsNode *> &GetChildren() const {
        assert(m_children.load() != nullptr && "MCTS NODE: DETECTED READ OF CHILDREN WITHOUT ASSIGNMENT!");
        return *m_children.load();
    }

    /**
     * @brief Increment given result atomically
     *
     * @param resultIdx index of result to increment
     *
     * @note index can be read from EVAL_RESULTS enum
     */
    void ScoreNode(const __uint32_t resultIdx) {
        m_scores[resultIdx].fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] __uint32_t GetScore(const __uint32_t idx) const {
        return m_scores[idx].load();
    }

    /**
     * @brief Clean up all child nodes
     */
    void Cleanup() {
        if (!m_children) {
            return;
        }

        while (!(m_children.load()->empty())) {
            const auto &child = m_children.load()->back();
            m_children.load()->pop_back();

            delete child;
        }

        delete m_children;
        m_children = nullptr;
    }

    /**
     * @brief Calculate Upper Confidence Bound (UCB) for node selection
     *
     * @return double UCB score for this node
     */
    [[nodiscard]] double CalculateUCB() {
        assert(m_parent != nullptr && "MCTS NODE UCB: FAILED NO PARENT");

        if (m_numSamples == 0) {
            return std::numeric_limits<double>::max();
        }

        const double averageScore = CalculateWinRate();
        const __uint64_t parentNumSamples = m_parent->GetNumSamples();

        return averageScore + UCB_COEF * std::sqrt(std::log(parentNumSamples) / m_numSamples);
    }

    /**
     * @brief Calculate win rate based on simulation scores
     *
     * @return double Estimated win probability for this node
     */
    [[nodiscard]] double CalculateWinRate() {
        const __uint64_t score = GetScore(m_board.MovingColor) + (GetScore(DRAW) + 1) / 2;
        return double(score) / double(m_numSamples);
    }

    // ------------------------------
    // Class implementation
    // ------------------------------
private:



    // ------------------------------
    // Class fields
    // ------------------------------
public:

    MctsNode *m_parent{};

    cuda_Move m_move{};
    cuda_Board m_board{};

private:
    std::atomic<std::vector<MctsNode *> *> m_children{};
    std::atomic<__uint32_t> m_scores[NUM_EVAL_RESULTS]{};
    std::atomic<__uint32_t> m_numSamples{};
};

#endif //SRC_MCTSNODE_CUH
