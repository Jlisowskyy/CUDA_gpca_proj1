//
// Created by Jlisowskyy on 19/11/24.
//

#include "CpuTests.h"

#include "engine/Checkmate-Chariot/include/MoveGeneration/MoveGenerator.h"
#include "engine/Checkmate-Chariot/include/Interface/FenTranslator.h"

#include <fstream>
#include <iostream>
#include <thread>
#include <tuple>
#include <mutex>

/**
 * @brief Simple pseudo-random number generator with XOR-shift algorithm.
 *
 * Modifies the input state using bitwise XOR operations to generate the next random value.
 *
 * @param state Reference to the random state, which is modified in-place
 */
void simpleRand(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
}

namespace cpu {

    uint64_t CountMoves(const external_board &board, int depth) {
        Board bd = TranslateToInternalBoard(board);

        Stack<Move, DEFAULT_STACK_SIZE> s;
        MoveGenerator mech{bd, s};

        return mech.CountMoves(bd, depth);
    }

    std::vector<std::string> LoadFenDb(const std::string &filename) {
        std::vector<std::string> results{};
        std::fstream file(filename, std::ios::in);

        if (!file.is_open()) {
            return results;
        }

        std::string line;
        while (std::getline(file, line)) {
            results.push_back(line);
        }

        file.close();
        return results;
    }

    static constexpr int MAX_THREADS = 20;
    inline static std::mutex mt{};

    void ThreadFunc(uint32_t idx, std::vector<Board> *boards, const std::vector<uint32_t> *seeds, uint32_t maxDepth,
                    uint32_t GPUthreads,
                    uint32_t retries, uint64_t *boardResults, uint64_t *moveResults) {
        Stack<Move, DEFAULT_STACK_SIZE> s;

        uint64_t boardsEvaluated{};
        uint64_t movesGenerated{};

        const uint32_t coef = GPUthreads / MAX_THREADS;
        for (uint32_t tries = 0; tries < retries; ++tries) {
            const uint32_t startIdx = idx * coef;
            const uint32_t stopIdx = (idx == MAX_THREADS - 1) ? GPUthreads : (idx + 1) * coef;

            for (uint32_t bidx = startIdx; bidx < stopIdx; ++bidx) {
                Board bd = (*boards)[bidx];
                uint32_t seed = (*seeds)[bidx];

                uint32_t depth{};
                while (depth < maxDepth) {
                    MoveGenerator mech{bd, s};
                    auto moves = mech.GetMovesFast();

                    movesGenerated += moves.size;
                    ++boardsEvaluated;

                    if (moves.size == 0) {
                        break;
                    }

                    Move pickedMove = moves[seed % moves.size];
                    Move::MakeMove(pickedMove, bd);

                    simpleRand(seed);
                    ++depth;

                    s.PopAggregate(moves);
                }
            }
        }

        mt.lock();
        *boardResults += boardsEvaluated;
        *moveResults += movesGenerated;
        mt.unlock();
    }

    std::tuple<double, uint64_t, uint64_t>
    TestMoveGenPerfCPU(const std::vector<std::string> &fens, uint32_t maxDepth, uint32_t GPUthreads, uint32_t retries,
                       const std::vector<uint32_t> &seeds) {
        std::cout << "Running Move Generation Performance Test on the CPU!" << std::endl;

        std::vector<Board> boards{};
        boards.resize(fens.size());

        for (size_t idx = 0; idx < fens.size(); ++idx) {
            const bool result = FenTranslator::Translate(fens[idx], boards[idx]);

            if (!result) {
                std::cout << "[ ERROR ] WRONG FEN DETECTED!" << std::endl;
            }
        }

        const auto t1 = std::chrono::high_resolution_clock::now();

        uint64_t boardsResults{};
        uint64_t moveResults{};
        std::vector<std::thread *> vThreads{};
        for (int idx = 0; idx < MAX_THREADS; ++idx) {
            vThreads.push_back(
                    new std::thread(&ThreadFunc, idx, &boards, &seeds, maxDepth, GPUthreads, retries, &boardsResults,
                                    &moveResults));
        }

        for (int idx = 0; idx < MAX_THREADS; ++idx) {
            vThreads[idx]->join();
            delete vThreads[idx];
        }

        const auto t2 = std::chrono::high_resolution_clock::now();
        const double seconds = std::chrono::duration<double>(t2 - t1).count();

        return {seconds, boardsResults, moveResults};
    }

}
