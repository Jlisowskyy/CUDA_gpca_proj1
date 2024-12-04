//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_CPUMOVEGEN_H
#define SRC_CPUMOVEGEN_H

#include "CpuDefines.h"

#include <vector>

class Board;

namespace cpu {

    [[nodiscard]] Board TranslateToInternalBoard(const external_board& exBoard);

    [[nodiscard]] uint64_t AccessCpuRookMap(int msbInd, uint64_t fullMap);

    [[nodiscard]] uint64_t AccessCpuBishopMap(int msbInd, uint64_t fullMap);

    [[nodiscard]] int ExtractMsbPosCPU(uint64_t map);

    [[nodiscard]] std::vector<external_move> GenerateMoves(const external_board &board);

    [[nodiscard]] bool IsCheck(const external_board &board);
}

#endif //SRC_CPUMOVEGEN_H
