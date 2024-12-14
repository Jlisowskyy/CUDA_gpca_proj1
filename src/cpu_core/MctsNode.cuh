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
#include <fstream>
#include <iostream>
#include <functional>
#include <iomanip>
#include <memory>

static constexpr double UCB_COEF = 1.0;
static constexpr double HONSETY_COEF = 0.02;

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
    [[nodiscard]] uint64_t GetNumSamples() const {
        return m_numSamples.load();
    }

    /**
     * @brief Increment number of simulation samples atomically
     */
    void IncNumSamples() {
        m_numSamples.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] uint32_t CalcDepth() const {
        uint32_t depth = 0;

        if (HasChildrenAssigned()) {
            for (const auto child: GetChildren()) {
                depth = std::max(depth, child->CalcDepth() + 1);
            }
        }

        return depth;
    }

    /**
     * @brief Perform compare and swap operation on children vector pointer, returns false when assignment failed
     *
     * @param children - pointer to vector of child nodes
     * @return bool - true if assignment was successful
     *
     * @note children pointer is expected to be allocated on heap, if failed nothing is done with the pointer
     */
    [[nodiscard("DATA LEAK POSSIBLE")]] bool SetChildren(std::vector<MctsNode *> *children) {
        assert(children != nullptr && "MCTS NODE: DETECTEC NULLPTR CHILDREN");

        std::vector<MctsNode *> *expected = nullptr;
        return m_children.compare_exchange_strong(expected, children);
    }

    /**
     * @brief Checks if children are assigned to this node
     *
     * @return bool - true if children are assigned
     */
    [[nodiscard]] bool HasChildrenAssigned() const {
        return m_children.load() != nullptr;
    }

    /**
     * @brief Get children of this node
     *
     * @return vector of child nodes
     *
     * @note before calling this method, make sure that children are assigned
     */
    [[nodiscard]] std::vector<MctsNode *> &GetChildren() const {
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
    void ScoreNode(const uint32_t resultIdx) {
        assert(resultIdx < 3 && "DETECTED WRONG RESULT IDX");
        m_scores[resultIdx].fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Get score of given result, performed atomically
     *
     * @param idx index of result to get
     * @return uint64_t score of given result
     */
    [[nodiscard]] uint64_t GetScore(const uint32_t idx) const {
        assert(idx < 3 && "DETECTED WRONG RESULT IDX");

        return m_scores[idx].load();
    }

    /**
     * @brief Clean up all child nodes
     */
    void Cleanup() {
        if (!m_children) {
            return;
        }

        for (auto &child: *m_children.load()) {
            delete child;
            child = nullptr;
        }

        delete m_children;
        m_children.store(nullptr);
    }

    /**
     * @brief Calculate Upper Confidence Bound (UCB) for node selection
     *
     * @return double UCB score for this node
     */
    [[nodiscard]] double CalculateUCB() const {
        assert(m_parent != nullptr && "MCTS NODE UCB: FAILED NO PARENT");

        if (GetNumSamples() == 0) {
            return std::numeric_limits<double>::max();
        }

        const double averageScore = CalculateWinRate();
        const uint64_t parentNumSamples = m_parent->GetNumSamples();
        const double evalScore = CalculateMaterialValue();

        return evalScore + averageScore + UCB_COEF *
               std::sqrt(std::log(parentNumSamples) / double(GetNumSamples()));
    }

    [[nodiscard]] double CalculateFinalWinRate() const {
        assert(m_numSamples.load() == (m_scores[0].load() + m_scores[1].load() + m_scores[2].load()) &&
            "MCTS NODE: DETECTED INCONSISTENT SAMPLE COUNT");

        return CalculateWinRate() + CalculateWinRateHonesty();
    }

    /**
     * @brief Calculate win rate based on simulation scores
     *
     * @return double Estimated win probability for this node
     */
    [[nodiscard]] double CalculateWinRate() const {
        assert(GetNumSamples() != 0 && "CALLED CALC WIN RATE WITHOUT ANY SAMPLES!");

        const uint64_t score = GetScore(SwapColor(m_board.MovingColor)) + (GetScore(DRAW) + 1) / 2;
        return double(score) / double(GetNumSamples());
    }

    [[nodiscard]] double CalculateWinRateHonesty() const {
        return HONSETY_COEF * (double(GetNumSamples()) / double(m_parent->GetNumSamples()));
    }

    [[nodiscard]] double CalculateMaterialValue() const {
        /* Note: value revers to parents perspective */
        const double materialEval = (m_board.MovingColor == BLACK ? 1.0 : -1.0) * double(m_board.MaterialEval);

        return materialEval / 10000.0;
    }

    /**
     * @brief Prepare graphical representation of the tree in DOT format
     *
     * @param filename - name of file to dump tree to
     */
    void DumpTreeToDotFormat(const std::string &filename, uint32_t maxDep = 10) {
        static constexpr uint64_t BUFF_SIZE = 200;

        std::ofstream dotFile(filename);
        if (!dotFile.is_open()) {
            std::cout << "Failed to open file for tree dump" << std::endl;
            return;
        }

        dotFile << "digraph OctTree {\n";
        dotFile << "    node [style=filled];\n";

        size_t nodeCounter = 0;
        auto label = new char[BUFF_SIZE];

        std::function<void(const MctsNode *, size_t, uint32_t)> traverseNode =
                [&](const MctsNode *currentNode, const size_t parentId, uint32_t depth) {
            if (!currentNode || depth > maxDep) {
                return;
            }

            const size_t currentNodeId = nodeCounter++;

            /* preventing usage of std::format due to GIANT compatibility issues */
            if (currentNode->m_parent) {
                snprintf(label, BUFF_SIZE,
                         "Move: %s\nColor:%d\nSamples: %lu\n[W WINS: %lu, B WINS: %lu, DRAWS: %lu]\nWinrate: %.5f\nUCB: %.5f\nEval points: %.5f\nEval: %d\nFinal eval: %.5f",
                         currentNode->m_move.GetPackedMove().GetLongAlgebraicNotationCPU().c_str(),
                         currentNode->m_board.MovingColor,
                         currentNode->GetNumSamples(),
                         currentNode->GetScore(WHITE),
                         currentNode->GetScore(BLACK),
                         currentNode->GetScore(DRAW),
                         currentNode->GetNumSamples() == 0 ? 0 : currentNode->CalculateWinRate(),
                         currentNode->CalculateUCB() == std::numeric_limits<double>::max()
                             ? 0
                             : currentNode->CalculateUCB(),
                         currentNode->CalculateMaterialValue(),
                         currentNode->m_board.MaterialEval,
                         currentNode->CalculateFinalWinRate()
                );
            } else {
                snprintf(label, BUFF_SIZE,
                         "PARENT node - no move\nSamples: %lu\n[W WINS: %lu, B WINS: %lu, DRAWS: %lu]\nWinrate: %.2f\n",
                         currentNode->GetNumSamples(),
                         currentNode->GetScore(WHITE),
                         currentNode->GetScore(BLACK),
                         currentNode->GetScore(DRAW),
                         currentNode->GetNumSamples() == 0 ? 0 : currentNode->CalculateWinRate()
                );
            }

            dotFile << "    node" << currentNodeId << " [label=\"" << label
                    << "\", fillcolor=\"#6fc787\"];\n";

            if (parentId != SIZE_MAX) {
                dotFile << "    node" << parentId << " -> node" << currentNodeId << ";\n";
            }

            if (!currentNode->HasChildrenAssigned()) {
                return;
            }

            for (const auto &child: currentNode->GetChildren()) {
                traverseNode(child, currentNodeId, depth + 1);
            }
        };

        traverseNode(this, SIZE_MAX, 0);

        delete label;
        dotFile << "}\n";
        dotFile.close();
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    MctsNode *m_parent{};

    cuda_Move m_move{};
    cuda_Board m_board{};

private:
    std::atomic<std::vector<MctsNode *> *> m_children{};
    std::atomic<uint64_t> m_scores[NUM_EVAL_RESULTS]{};
    std::atomic<uint64_t> m_numSamples{};
};

#endif //SRC_MCTSNODE_CUH
