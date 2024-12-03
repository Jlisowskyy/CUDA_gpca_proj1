//
// Created by Jlisowskyy on 19/11/24.
//

#ifndef SRC_CPUUTILS_H
#define SRC_CPUUTILS_H

#include "CpuDefines.h"

#include <string>

namespace cpu {

    [[nodiscard]] external_board TranslateFromFen(const std::string &fen);

    [[nodiscard]] std::string TranslateToFen(const external_board& board);

    [[nodiscard]] external_board GetDefaultBoard();

    [[nodiscard]] bool Translate(const std::string &fen, external_board &board);
}

#endif //SRC_CPUUTILS_H
