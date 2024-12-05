//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CLI_H
#define SRC_CLI_H

/* Forward declaration */
class CpuCore;

struct cuda_Board;

class Cli final {
public:
    // ------------------------------
    // internal types
    // ------------------------------

    enum class RC_BoardLoad {
        SUCCESS,
        FAILURE,
        EXIT,
    };

    enum class RC_GameTypeLod {
        COMPUTER_VS_COMPUTER,
        PLAYER_VS_COMPUTER,
        FAILURE,
        TEST,
        EXIT,
        INFINITE,
    };

    // ------------------------------
    // Class creation
    // ------------------------------

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

    [[nodiscard]] RC_BoardLoad _loadPosition(cuda_Board &board) const;

    [[nodiscard]] RC_GameTypeLod _loadGameType() const;

    void _runGame(Cli::RC_GameTypeLod gameType);

    [[nodiscard]] static __uint32_t _readSecondsPerMove();

    [[nodiscard]] static __uint32_t _readPlayingColor();

    // ------------------------------
    // Class fields
    // ------------------------------

    CpuCore *m_core{};
    cuda_Board *m_board;
};


#endif //SRC_CLI_H
