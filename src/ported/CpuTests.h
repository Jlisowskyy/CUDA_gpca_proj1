//
// Created by Jlisowskyy on 17/11/24.
//

#ifndef SRC_CPUTESTS_H
#define SRC_CPUTESTS_H

#include <cstdint>
#include <string>
#include <vector>
#include <tuple>

#include "CpuDefines.h"
#include "CpuMoveGen.h"

namespace cpu {

    void FancyMagicTest();

    /**
     * @brief Read and parse magic bitboard correctness test records from a file.
     *
     * @param filename Path to the test records file
     * @return MapCorrecntessRecordsPack Tuple containing test records and metadata
     */
    MapCorrecntessRecordsPack ReadMagicCorrectnessTestFile(const std::string &filename);

    /**
     * @brief Read and parse magic bitboard performance test records from a file.
     *
     * @param filename Path to the test records file
     * @return MapPerformanceRecordsPack Tuple containing performance test records
     */
    MapPerformanceRecordsPack ReadMagicPerformanceTestFile(const std::string &filename);

    /**
     * @brief Display a bitboard's contents in a visual representation on the CPU.
     *
     * @param board 64-bit bitboard to be displayed
     */
    void DisplayBitBoardCPU(uint64_t board);

    /**
     * @brief Count the number of moves possible from a given board position to a specified depth.
     *
     * This function translates an external board representation to an internal board,
     * then uses the MoveGenerator to count all possible moves recursively up to the specified depth.
     *
     * @param board External board representation to analyze
     * @param depth Maximum depth to generate moves
     * @return uint64_t Total number of moves generated across all levels
     */
    uint64_t CountMoves(const external_board &board, int depth);

    /**
     * @brief Load FEN strings from a given file.
     *
     * Reads a text file containing FEN strings, with one FEN per line.
     * Returns an empty vector if the file cannot be opened.
     *
     * @param filename Path to the file containing FEN strings
     * @return std::vector<std::string> Vector of FEN strings read from the file
     */
    std::vector<std::string> LoadFenDb(const std::string &filename);

    /**
     * @brief Perform a multi-threaded move generation performance test on CPU.
     *
     * Generates moves for multiple board positions across multiple threads to benchmark
     * move generation performance. Distributes work across threads based on GPU thread count.
     *
     * @param fens Vector of FEN strings representing initial board positions
     * @param maxDepth Maximum move generation depth
     * @param GPUthreads Total number of GPU threads to simulate
     * @param retries Number of test iterations
     * @param seeds Random seeds for move selection
     * @return std::tuple<double, uint64_t, uint64_t> Tuple containing:
     *         - Execution time in seconds
     *         - Total boards evaluated
     *         - Total moves generated
     */
    std::tuple<double, uint64_t, uint64_t>
    TestMoveGenPerfCPU(const std::vector<std::string> &fens, uint32_t maxDepth, uint32_t GPUthreads, uint32_t retries,
                       const std::vector<uint32_t> &seeds);
} // namespace cpu


#endif //SRC_CPUTESTS_H
