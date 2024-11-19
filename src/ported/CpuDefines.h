//
// Created by Jlisowskyy on 19/11/24.
//

#ifndef SRC_CPUDEFINES_H
#define SRC_CPUDEFINES_H

#include <tuple>
#include <cstdint>
#include <vector>
#include <array>

namespace cpu {

    using MapCorrecntessRecordsPack =
            std::tuple<uint64_t, std::vector<uint64_t>, std::vector<uint64_t>, std::vector<std::vector<uint64_t>>>;

    using MapPerformanceRecordsPack = std::tuple<uint64_t, std::vector<uint64_t>, std::vector<uint64_t>>;

    using external_move = std::array<uint16_t, 3>;
    using external_board = std::array<uint64_t, 15>;

}

#endif //SRC_CPUDEFINES_H
