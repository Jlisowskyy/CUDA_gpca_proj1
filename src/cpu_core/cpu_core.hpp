//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CPU_CORE_H
#define SRC_CPU_CORE_H

#include "../data_structs/Board.hpp"

class cpu_core final {
    // ------------------------------
    // Class creation
    // ------------------------------
public:

    cpu_core();
    ~cpu_core();

    // ------------------------------
    // Class interaction
    // ------------------------------

    void setBoard(const Board &board) {
        m_board = board;
    }

    void runCVC();

    void runPVC();

    void init();

    // ------------------------------
    // Class fields
    // ------------------------------
private:

    Board m_board;
};


#endif //SRC_CPU_CORE_H
