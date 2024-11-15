//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CLI_H
#define SRC_CLI_H

/* Forward declaration */
class CpuCore;

struct cpu_Board;

class Cli final {
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

    explicit Cli(CpuCore *core);

    ~Cli();

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

    [[nodiscard]] RC_BoardLoad _loadPosition(cpu_Board &board) const;

    [[nodiscard]] RC_GameTypeLod _loadGameType() const;

    void _runGame(Cli::RC_GameTypeLod gameType);

    // ------------------------------
    // Class fields
    // ------------------------------

    CpuCore *m_core{};
    cpu_Board *m_board;
};


#endif //SRC_CLI_H
