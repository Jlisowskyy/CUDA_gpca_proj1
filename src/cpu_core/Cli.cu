//
// Created by Jlisowskyy on 14/11/24.
//

#include "Cli.cuh"

#include <iostream>
#include <string>
#include <string_view>
#include <cassert>

#include "../cpu_core/CpuCore.cuh"
#include "../cpu_core/TestRunner.cuh"
#include "../cuda_core/cuda_Board.cuh"

#include "../ported/CpuUtils.h"

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
        case RC_GameTypeLod::COMPUTER_VS_COMPUTER:
            m_core->runCVC();
            break;
        case RC_GameTypeLod::PLAYER_VS_COMPUTER:
            m_core->runPVC();
            break;
        case RC_GameTypeLod::TEST:
            TestRunner(m_core).runTests();
            break;
        default:
            assert(false && "Invalid game type provided!");
    }
}

Cli::RC_GameTypeLod Cli::_loadGameType() const {
    std::string input;
    std::getline(std::cin, input);

    if (input == "exit") {
        std::cout << "Exiting game!" << std::endl;
        return RC_GameTypeLod::EXIT;
    }

    if (input == "cvc") {
        std::cout << "Computer vs Computer game started!" << std::endl;
        return RC_GameTypeLod::COMPUTER_VS_COMPUTER;
    }

    if (input == "pvc") {
        std::cout << "Player vs Computer game started!" << std::endl;
        return RC_GameTypeLod::PLAYER_VS_COMPUTER;
    }

    if (input == "test") {
        std::cout << "Entering dev tes mode!" << std::endl;
        return RC_GameTypeLod::TEST;
    }

    std::cout << "Invalid game type provided!" << std::endl;
    return Cli::RC_GameTypeLod::FAILURE;
}

void Cli::_displayGameTypeMessage() {
    static constexpr std::string_view gameTypeMsg = R"(
Choose game type:
- 'cvc' for Computer vs Computer
- 'pvc' for Player vs Computer

Please provide input:)";

    std::cout << gameTypeMsg << std::endl;
}