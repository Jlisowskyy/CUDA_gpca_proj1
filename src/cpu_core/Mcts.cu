//
// Created by Jlisowskyy on 05/12/24.
//

#include "Mcts.cuh"

#include "cpu_MoveGen.cuh"

#include <numeric>

namespace mcts {
    std::atomic<__uint32_t> g_ExpandRacesCounter{};
    std::atomic<__uint64_t> g_SimulationCounter{};

    void ExpandTreeCPU(MctsNode *root) {
        throw std::runtime_error("NOT IMPLEMENTED");
    }

    MctsNode *SelectNode(MctsNode *const root) {
        assert(root != nullptr && "NULLPTR NODE DETECTED!");

        root->IncNumSamples();
        if (!root->HasChildrenAssigned()) {
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

        return root->GetChildren()[root->GetNumSamples() % root->GetChildren().size()];
    }

    void PropagateResult(MctsNode *const node, const __uint32_t result) {
        if (node == nullptr) {
            return;
        }

        node->ScoreNode(result);
        PropagateResult(node->m_parent, result);
    }
}
