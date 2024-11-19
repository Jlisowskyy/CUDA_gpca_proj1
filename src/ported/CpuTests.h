//
// Created by Jlisowskyy on 17/11/24.
//

#ifndef SRC_CPUTESTS_H
#define SRC_CPUTESTS_H

#include <cstdint>
#include <string>
#include <vector>

#include "CpuDefines.h"

namespace cpu {

    void FancyMagicTest();

    MapCorrecntessRecordsPack ReadMagicCorrectnessTestFile(const std::string& filename);

    MapPerformanceRecordsPack ReadMagicPerformanceTestFile(const std::string& filename);

    uint64_t AccessCpuRookMap(int msbInd, uint64_t fullMap);

    uint64_t AccessCpuBishopMap(int msbInd, uint64_t fullMap);

    int ExtractMsbPosCPU(uint64_t map);

    void DisplayBoardCPU(uint64_t board);

    std::vector<external_move> GenerateMoves(const external_board& board);
} // namespace cpu


#endif //SRC_CPUTESTS_H
