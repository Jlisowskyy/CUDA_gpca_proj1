//
// Created by Jlisowskyy on 14/11/24.
//

#include "Cli.cuh"

#include <iostream>
#include <string>
#include <string_view>
#include <cassert>
#include <unordered_map>

#include "../cpu_core/CpuCore.cuh"
#include "../cpu_core/TestRunner.cuh"
#include "../cuda_core/cuda_Board.cuh"

#include "../ported/CpuUtils.h"

#include "Utils.cuh"

Cli::~Cli() {
    delete m_board;
}

void Cli::run() {
    _displayWelcomeMessage();

    RC_BoardLoad rc;
    do {
        rc = _loadPosition(*m_board);
    } while (rc == RC_BoardLoad::FAILURE);

    if (rc == RC_BoardLoad::EXIT) {
        return;
    }

    _displayGameTypeMessage();

    RC_GameTypeLod rc1;
    do {
        rc1 = _loadGameType();
    } while (rc1 == RC_GameTypeLod::FAILURE);

    if (rc1 == RC_GameTypeLod::EXIT) {
        return;
    }

    _runGame(rc1);
}

Cli::Cli(CpuCore *core) : m_core(core), m_board(new cuda_Board()) {
    assert(m_core != nullptr && "Core must be provided!");
}

Cli::RC_BoardLoad Cli::_loadPosition(cuda_Board &board) const {
    std::string input;
    std::getline(std::cin, input);

    if (input == "exit") {
        std::cout << "Exiting game!" << std::endl;
        return RC_BoardLoad::EXIT;
    }

    if (input.empty()) {
        std::cout << "Loading default position!" << std::endl;
        board = cuda_Board(cpu::GetDefaultBoard());
        return RC_BoardLoad::SUCCESS;
    }

    cpu::external_board eBd{};
    if (!cpu::Translate(input, eBd)) {
        std::cout << "Invalid FEN string provided!" << std::endl;
        return RC_BoardLoad::FAILURE;
    } else {
        board = cuda_Board(eBd);
    }

    return RC_BoardLoad::SUCCESS;
}

void Cli::_displayWelcomeMessage() {
    static constexpr std::string_view welcomeMsg = R"(
Welcome to Checkmate-Chariot-MCTS-Cuda!
Simplistic Chess Engine written for University project for
Graphic Processors in Computational Application Course!

Provide FEN string to start the game or type 'exit' to quit.
Simply pressing enter will input default position.

Provide input:)";

    std::cout << welcomeMsg << std::endl;
}

void Cli::_runGame(Cli::RC_GameTypeLod gameType) {
    m_core->setBoard(m_board);

    switch (gameType) {
        case RC_GameTypeLod::COMPUTER_VS_COMPUTER_BEST_GPU:
            m_core->runCVC(_readSecondsPerMove());
            break;
        case RC_GameTypeLod::PLAYER_VS_COMPUTER:
            m_core->runPVC(_readSecondsPerMove(), _readPlayingColor());
            break;
        case RC_GameTypeLod::TEST:
            TestRunner(m_core).runTests();
            break;
        case RC_GameTypeLod::INFINITE:
            m_core->runInfinite();
            break;
        case RC_GameTypeLod::COMPUTER_VS_COMPUTER_CPU_VS_BEST_GPU:
            m_core->runCVC1(_readSecondsPerMove());
            break;
        case RC_GameTypeLod::COMPUTER_VS_COMPUTER_GPU0_VS_GPU1:
            m_core->runCVC2(_readSecondsPerMove());

            break;
        default:
            assert(false && "Invalid game type provided!");
    }
}

Cli::RC_GameTypeLod Cli::_loadGameType() const {
    static std::unordered_map<std::string, RC_GameTypeLod> inputMap{
            {"exit",     RC_GameTypeLod::EXIT},
            {"cvc",  RC_GameTypeLod::COMPUTER_VS_COMPUTER_BEST_GPU},
            {"pvc",      RC_GameTypeLod::PLAYER_VS_COMPUTER},
            {"test",     RC_GameTypeLod::TEST},
            {"infinite", RC_GameTypeLod::INFINITE},
            {"cvc1", RC_GameTypeLod::COMPUTER_VS_COMPUTER_CPU_VS_BEST_GPU},
            {"cvc2", RC_GameTypeLod::COMPUTER_VS_COMPUTER_GPU0_VS_GPU1}
    };

    static std::unordered_map<RC_GameTypeLod, std::string_view> messages{
            {RC_GameTypeLod::EXIT,                                 "Exiting game!"},
            {RC_GameTypeLod::COMPUTER_VS_COMPUTER_BEST_GPU,        "Computer vs Computer game started!"},
            {RC_GameTypeLod::PLAYER_VS_COMPUTER,                   "Player vs Computer game started!"},
            {RC_GameTypeLod::TEST,                                 "Entering dev tes mode!"},
            {RC_GameTypeLod::INFINITE,                             "Starting infinite search. Type \"stop\" to end the program"},
            {RC_GameTypeLod::COMPUTER_VS_COMPUTER_GPU0_VS_GPU1,    "Computer vs Computer game started!"},
            {RC_GameTypeLod::COMPUTER_VS_COMPUTER_CPU_VS_BEST_GPU, "Computer vs Computer game started!"}
    };

    std::string input;
    std::getline(std::cin, input);
    cpu::Trim(input);
    StrToLower(input);

    if (inputMap.find(input) == inputMap.end()) {
        std::cout << "Invalid game type provided!" << std::endl;
        return Cli::RC_GameTypeLod::FAILURE;
    }

    RC_GameTypeLod type = inputMap[input];
    std::cout << messages[type] << std::endl;
    return type;
}

void Cli::_displayGameTypeMessage() {
    static constexpr std::string_view gameTypeMsg = R"(
Choose game type:
- 'cvc'      for Computer vs Computer (Best GPU vs Best GPU)
- 'cvc1'     for Computer vs Computer (Best GPU vs CPU)
- 'cvc2'     for Computer vs Computer (GPU0 (BEST) vs GPU1)
- 'pvc'      for Player vs Computer
- 'infinite' for infinite search on given position
- 'test'     to enter test suite

Please provide input:)";

    std::cout << gameTypeMsg << std::endl;
}

__uint32_t Cli::_readSecondsPerMove() {
    static constexpr std::string_view msg = "Provide time for the engine too analyze single move (MILLISECONDS):";

    std::string input{};
    __uint32_t result{};

    do {
        std::cout << msg << std::endl;
        std::getline(std::cin, input);

        try {
            result = std::stoul(input);
        } catch (...) {
            result = 0;
        }
    } while (result == 0);

    return result;
}

__uint32_t Cli::_readPlayingColor() {
    static constexpr std::string_view msg = "Pick your color to play: (white, black, b, w)";
    static std::unordered_map<std::string, __uint32_t> inputMap{
            {"white", WHITE},
            {"w",     WHITE},
            {"black", BLACK},
            {"b",     BLACK},
    };

    std::string input{};
    do {
        std::cout << msg << std::endl;

        std::getline(std::cin, input);
        cpu::Trim(input);
        StrToLower(input);

    } while (inputMap.find(input) == inputMap.end());

    return inputMap[input];
}
