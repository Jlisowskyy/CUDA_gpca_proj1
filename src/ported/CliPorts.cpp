//
// Created by Jlisowskyy on 19/11/24.
//

#include "CpuUtils.h"

#include "../../engine/include/Interface/FenTranslator.h"
#include "../../engine/include/ParseTools.h"

namespace cpu {
    external_board TranslateFromFen(const std::string &fen) {
        external_board eBd{};

        const Board bd = FenTranslator::GetTranslated(fen);

        for (size_t i = 0; i < Board::BitBoardsCount; ++i) {
            eBd[i] = bd.BitBoards[i];
        }

        eBd[12] = bd.ElPassantField;
        eBd[13] = bd.Castlings.to_ullong();
        eBd[14] = bd.MovingColor;

        return eBd;
    }

    external_board GetDefaultBoard() {
        external_board eBd{};

        const Board bd = FenTranslator::GetDefault();

        for (size_t i = 0; i < Board::BitBoardsCount; ++i) {
            eBd[i] = bd.BitBoards[i];
        }

        eBd[12] = bd.ElPassantField;
        eBd[13] = bd.Castlings.to_ullong();
        eBd[14] = bd.MovingColor;

        return eBd;
    }

    bool Translate(const std::string &fen, external_board &board) {
        Board bd{};

        if (!FenTranslator::Translate(fen, bd)) {
            return false;
        }

        external_board eBd{};
        for (size_t i = 0; i < Board::BitBoardsCount; ++i) {
            eBd[i] = bd.BitBoards[i];
        }

        eBd[12] = bd.ElPassantField;
        eBd[13] = bd.Castlings.to_ullong();
        eBd[14] = bd.MovingColor;

        board = eBd;
        return true;
    }

    std::string TranslateToFen(const external_board &board) {
        Board bd{};

        for (size_t i = 0; i < Board::BitBoardsCount; ++i) {
            bd.BitBoards[i] = board[i];
        }

        // Copy special fields
        bd.ElPassantField = board[12];
        bd.Castlings = std::bitset<Board::CastlingCount + 1>(board[13]);
        bd.MovingColor = static_cast<int>(board[14]);

        return FenTranslator::Translate(bd);
    }

    void Trim(std::string &str) {
        str = ParseTools::GetTrimmed(str);
    }
}