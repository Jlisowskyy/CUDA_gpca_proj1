//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CLI_H
#define SRC_CLI_H

/* Forward declaration */
class CpuCore;

struct cuda_Board;

/**
 * @class Cli
 * @brief Command-line interface for managing chess game interactions
 *
 * Provides a text-based interface for:
 * - Loading chess board positions
 * - Selecting game modes
 * - Configuring game parameters
 * - Initiating different types of chess game scenarios
 */
class Cli final {
public:
    // ------------------------------
    // internal types
    // ------------------------------

    /**
     * @enum RC_BoardLoad
     * @brief Return codes for board loading process
     */
    enum class RC_BoardLoad {
        SUCCESS,
        FAILURE,
        EXIT,
    };

    /**
     * @enum RC_GameTypeLod
     * @brief Return codes for game type selection
     */
    enum class RC_GameTypeLod {
        COMPUTER_VS_COMPUTER_BEST_GPU,
        PLAYER_VS_COMPUTER,
        FAILURE,
        TEST,
        EXIT,
        INFINITE,
        COMPUTER_VS_COMPUTER_CPU_VS_BEST_GPU,
        COMPUTER_VS_COMPUTER_GPU0_VS_GPU1,
    };

    // ------------------------------
    // Class creation
    // ------------------------------

    /**
     * @brief Construct a new Cli object
     *
     * @param core Pointer to CpuCore for game processing
     * @pre core must not be nullptr
     */
    explicit Cli(CpuCore *core);

    ~Cli();

    // ------------------------------
    // Class interaction
    // ------------------------------

    /**
     * @brief Main entry point for CLI application
     *
     * Manages game setup, board loading, and game mode selection
     */
    void run();

    // ------------------------------
    // class private methods
    // ------------------------------
private:

    /**
     * @brief Displays initial welcome message
     */
    static void _displayWelcomeMessage();

    /**
     * @brief Displays available game type options
     */
    static void _displayGameTypeMessage();

    /**
     * @brief Load chess board position from user input
     *
     * @param board Reference to board to be loaded
     * @return RC_BoardLoad Status of board loading
     */
    [[nodiscard]] RC_BoardLoad _loadPosition(cuda_Board &board) const;

    /**
     * @brief Prompt user to select game type
     *
     * @return RC_GameTypeLod Selected game type
     */
    [[nodiscard]] RC_GameTypeLod _loadGameType() const;

    /**
     * @brief Initialize and run selected game mode
     *
     * @param gameType Chosen game mode
     */
    void _runGame(Cli::RC_GameTypeLod gameType);

    /**
     * @brief Read move thinking time from user
     *
     * @return __uint32_t Time in milliseconds
     */
    [[nodiscard]] static __uint32_t _readSecondsPerMove();

    /**
     * @brief Prompt user to select playing color
     *
     * @return __uint32_t Selected color (WHITE or BLACK)
     */

    [[nodiscard]] static __uint32_t _readPlayingColor();

    // ------------------------------
    // Class fields
    // ------------------------------

    CpuCore *m_core{};
    cuda_Board *m_board;
};


#endif //SRC_CLI_H
