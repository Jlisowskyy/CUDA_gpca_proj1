//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CPU_CORE_H
#define SRC_CPU_CORE_H

struct cudaDeviceProp;
struct cuda_Board;

#include <tuple>
#include "../cuda_core/Move.cuh"
#include "MctsEngine.cuh"

/**
 * @class CpuCore
 * @brief Manages CPU-side chess engine operations, GPU device selection, and game modes
 *
 * The CpuCore class provides a comprehensive interface for running chess games
 * using CUDA-accelerated move generation and Monte Carlo Tree Search (MCTS) engine.
 * It supports multiple game modes including:
 * - Computer vs Computer (CVC)
 * - Player vs Computer (PVC)
 * - Infinite search mode
 *
 * Key responsibilities include:
 * - GPU device selection and initialization
 * - Board state management
 * - Game mode execution
 */
class CpuCore final {
    // ------------------------------
    // Class creation
    // ------------------------------
public:

    /**
     * @brief Default constructor for CpuCore
     *
     * Initializes an empty CpuCore instance without setting up a board or GPU
     */
    CpuCore();

    ~CpuCore();

    // ------------------------------
    // Class interaction
    // ------------------------------

    /**
     * @brief Sets the chess board state for the current game
     *
     * @param board Pointer to a cuda_Board representing the initial game state
     * @pre board must not be nullptr
     */
    void setBoard(cuda_Board *board);

    /**
     * @brief Runs a Computer vs Computer (CVC) game mode
     *
     * @param moveTime Maximum thinking time for each engine move in milliseconds
     */
    void runCVC(uint32_t moveTime);

    /**
     * @brief Runs a Computer vs Computer (CVC) game mode (BEST GPU VS CPU)
     *
     * @param moveTime Maximum thinking time for each engine move in milliseconds
     */
    void runCVC1(uint32_t moveTime);

    /**
     * @brief Runs a Computer vs Computer (CVC) game mode (GPU0 VS GPU1)
     *
     * @param moveTime Maximum thinking time for each engine move in milliseconds
     */
    void runCVC2(uint32_t moveTime);

    /**
     * @brief Runs a Player vs Computer (PVC) game mode
     *
     * @param moveTime Maximum thinking time for the computer's moves
     * @param playerColor Color of the human player (WHITE or BLACK)
     */
    void runPVC(uint32_t moveTime, uint32_t playerColor);

    /**
     * @brief Runs the engine in infinite search mode
     *
     * Continuously searches for the best move until manually stopped
     * Waits for a "stop" command from user input
     */
    void runInfinite();

    /**
     * @brief Initializes the CpuCore by selecting and configuring the best GPU
     *
     * Performs GPU device detection, selects the most suitable device,
     * and sets up necessary CUDA configurations
     */
    void init();

    /**
     * @brief Returns the number of device threads available
     *
     * @return int Number of hardware threads on the selected GPU
     */
    [[nodiscard]] int getDeviceThreads() const {
        return m_deviceThreads;
    }

    [[nodiscard]] const cudaDeviceProp &getDeviceProps() const {
        return *m_deviceProps;
    }

    // ------------------------------
    // Class private methods
    // ------------------------------
private:

    /**
     * @brief Selects the best available CUDA-capable GPU device, based on the number of available threads
     *
     * @return std::tuple containing:
     *         - Best device index
     *         - Number of hardware threads
     *         - Pointer to device properties
     */
    [[nodiscard]] std::tuple<int, int, cudaDeviceProp *> _pickGpu();

    /**
     * @brief Prints detailed information about a CUDA device
     *
     * @param idx Device index
     * @param props Device properties to display
     */
    void _dumpGPUInfo(int idx, const cudaDeviceProp &props);

    /**
     * @brief Reads a player's move from console input
     *
     * @param correctMoves Vector of legal moves for validation
     * @return cuda_Move Selected and validated player move
     */
    [[nodiscard]] static cuda_Move _readPlayerMove(const std::vector<cuda_Move>& correctMoves);

    /**
     * @brief Validates if a move is legal
     *
     * @param validMoves Vector of legal moves
     * @param move Move to validate
     * @return bool True if the move is legal, false otherwise
     */
    [[nodiscard]] static bool _validateMove(const std::vector<cuda_Move>& validMoves, cuda_Move move);

    /**
     * @brief Runs a progress bar animation during move search
     *
     * @param moveTime Duration of the move search in milliseconds
     *
     * @note Also stands as blocking mechanism to pass the time
     */
    static void _runProcessingAnim(uint32_t moveTime);

    /**
     * @brief Waits for user input to stop infinite search mode
     *
     * Blocks execution until "stop" command is received
     */
    static void _waitForInfiniteStop(MctsEngine <EngineType::GPU1> &engine);

    template<class ENGINE_T1, class ENGINE_T2>
    void _runCVC(uint32_t moveTime);

    // ------------------------------
    // Class fields
    // ------------------------------

    cuda_Board *m_board{};

    int m_deviceThreads{};

    cudaDeviceProp *m_deviceProps{};
};


#endif //SRC_CPU_CORE_H
