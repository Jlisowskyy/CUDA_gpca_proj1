//
// Created by Jlisowskyy on 05/12/24.
//

#include "Mcts.cuh"

#include "cpu_MoveGen.cuh"

#include <numeric>

namespace mcts {
    std::atomic<uint32_t> g_ExpandRacesCounter{};
    std::atomic<uint64_t> g_SimulationCounter{};
    std::atomic<double> g_CopyTimes{};
    std::atomic<double> g_KernelTime{};
    std::atomic<double> g_CopyBackTimes{};

    void ExpandTreeCPU(MctsNode *root) {
        MctsNode *node = SelectNode(root);

        /* We selected node already extended but without any children roll back */
        if (node->HasChildrenAssigned()) {
            PropagateResult(node,
                            ported_translation::IsCheck(node->m_board) ?
                            SwapColor(node->m_board.MovingColor) : DRAW
            );
            g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
            return;
        }

        MctsNode *expandedNode = ExpandNode(node);

        /* Selected node was not expanded yet, but we found out that it is a dead end indeed */
        if (expandedNode == nullptr) {
            PropagateResult(node,
                            ported_translation::IsCheck(node->m_board) ?
                            SwapColor(node->m_board.MovingColor) : DRAW
            );
            g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
            return;
        }

        const uint32_t result = cpu::SimulateGame(expandedNode->m_board.DumpToExternal());
        PropagateResult(expandedNode, result);

        g_SimulationCounter.fetch_add(1, std::memory_order::relaxed);
    }

    MctsNode *SelectNode(MctsNode *const root) {
        assert(root != nullptr && "NULLPTR NODE DETECTED!");

        root->IncNumSamples();

        /* if we have no children return ourselves and let expand fail */
        /* NOTE: after selection there should be check if return node has children or not! */
        if (!root->HasChildrenAssigned() || root->GetChildren().empty()) {
            return root;
        }

        double bestScore = std::numeric_limits<double>::min();
        MctsNode *node{};

        for (const auto child: root->GetChildren()) {
            if (const double score = child->CalculateUCB(); score > bestScore) {
                bestScore = score;
                node = child;
            }
        }

        return SelectNode(node);
    }

    MctsNode *ExpandNode(MctsNode *const root) {
        if (root->GetNumSamples() < MIN_SAMPLES_TO_EXPAND) {
            return root;
        }

        /* generate all possible children */
        const auto moves = ported_translation::GenMoves(root->m_board);

        /* Alloc the bector */
        const auto pChildren = new std::vector<MctsNode *>();
        pChildren->reserve(moves.size());

        /* Add child nodes */
        for (const auto move: moves) {
            pChildren->emplace_back(new MctsNode(root, move));
        }

        const bool result = root->SetChildren(pChildren);

        if (!result) {
            g_ExpandRacesCounter.fetch_add(1, std::memory_order_relaxed);

            for (const auto &child: *pChildren) {
                delete child;
            }
            delete pChildren;
        }

        return root->GetChildren().empty() ? nullptr :
               root->GetChildren()[root->GetNumSamples() % root->GetChildren().size()];
    }

    void PropagateResult(MctsNode *const node, const uint32_t result) {
        if (node == nullptr) {
            return;
        }

        node->ScoreNode(result);
        PropagateResult(node->m_parent, result);
    }

}
