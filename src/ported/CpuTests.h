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

    MapCorrecntessRecordsPack ReadMagicCorrectnessTestFile(const std::string &filename);

    MapPerformanceRecordsPack ReadMagicPerformanceTestFile(const std::string &filename);

    void DisplayBitBoardCPU(uint64_t board);

    uint64_t CountMoves(const external_board &board, int depth);

    std::vector<std::string> LoadFenDb(const std::string &filename);

    std::tuple<double, uint64_t, uint64_t>
    TestMoveGenPerfCPU(const std::vector<std::string> &fens, uint32_t maxDepth, uint32_t GPUthreads, uint32_t retries,
                       const std::vector<uint32_t> &seeds);
} // namespace cpu


#endif //SRC_CPUTESTS_H
