//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CLI_H
#define SRC_CLI_H

/* Forward declaration */
class cpu_core;

#include "../data_structs/Board.hpp"

class cli final {
    // ------------------------------
    // internal types
    // ------------------------------

    enum class RC_BoardLoad {
        SUCCESS,
        FAILURE,
        EXIT
    };

    enum class RC_GameTypeLod {
        COMPUTER_VS_COMPUTER,
        PLAYER_VS_COMPUTER,
        FAILURE,
        EXIT
    };

    // ------------------------------
    // Class creation
    // ------------------------------
public:

    explicit cli(cpu_core* core);
    ~cli();

    // ------------------------------
    // Class interaction
    // ------------------------------

    void run();

    // ------------------------------
    // class private methods
    // ------------------------------
private:

    static void _displayWelcomeMessage();

    static void _displayGameTypeMessage();

    [[nodiscard]] RC_BoardLoad _loadPosition(Board &board) const;

    [[nodiscard]] RC_GameTypeLod _loadGameType() const;

    void _runGame(cli::RC_GameTypeLod gameType);

    // ------------------------------
    // Class fields
    // ------------------------------

    cpu_core* m_core{};
    Board m_board{};
};


#endif //SRC_CLI_H
