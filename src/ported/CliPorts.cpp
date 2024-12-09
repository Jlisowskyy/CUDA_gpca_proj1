//
// Created by Jlisowskyy on 19/11/24.
//

#include "CpuUtils.h"
#include "CpuMoveGen.h"

#include "engine/Checkmate-Chariot/include/Interface/FenTranslator.h"
#include "engine/Checkmate-Chariot/include/ParseTools.h"

static cpu::external_board TraslateToExternal(const Board &bd) {
    cpu::external_board eBd{};

    for (size_t i = 0; i < Board::BitBoardsCount; ++i) {
        eBd[i] = bd.BitBoards[i];
    }

    eBd[12] = bd.ElPassantField;
    eBd[13] = bd.Castlings.to_ullong();
    eBd[14] = bd.MovingColor;
    eBd[15] = bd.HalfMoves;

    return eBd;
}

namespace cpu {
    external_board TranslateFromFen(const std::string &fen) {
        return TraslateToExternal(FenTranslator::GetTranslated(fen));
    }

    external_board GetDefaultBoard() {
        return TraslateToExternal(FenTranslator::GetDefault());
    }

    bool Translate(const std::string &fen, external_board &board) {
        Board bd{};

        if (!FenTranslator::Translate(fen, bd)) {
            return false;
        }

        board = TraslateToExternal(bd);
        return true;
    }

    std::string TranslateToFen(const external_board &board) {
        const Board bd = TranslateToInternalBoard(board);
        return FenTranslator::Translate(bd);
    }

    void Trim(std::string &str) {
        str = ParseTools::GetTrimmed(str);
    }

    void DisplayBoard(const external_board &board) {
        const Board bd = TranslateToInternalBoard(board);
        DisplayBoard(bd);
        std::cout << "Position: " << FenTranslator::Translate(bd) << std::endl;
    }
}
