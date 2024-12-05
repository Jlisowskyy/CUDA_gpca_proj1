//
// Created by Jlisowskyy on 05/12/24.
//

#include "Mcts.cuh"

#include <numeric>

namespace mcts {
    void ExpandTreeCPU(MctsNode *root) {
        throw std::runtime_error("NOT IMPLEMENTED");
    }

    MctsNode *SelectNode(MctsNode *root) {
        assert(root != nullptr && "NULLPTR NODE DETECTED!");

        double bestScore = std::numeric_limits<double>::max();
        MctsNode *node{};

        for (const auto child: root->m_children) {
            if (const double score = child->CalculateUCB(); score > bestScore) {
                bestScore = score;
                node = child;
            }
        }

        root->IncNumSamples();
        if (node) {
            return SelectNode(node);
        }

        return root;
    }
}
