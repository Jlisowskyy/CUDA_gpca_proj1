//
// Created by Jlisowskyy on 12/31/23.
//

#ifndef CHESSMECHANICS_H
#define CHESSMECHANICS_H

#include "../../utilities/BitOperations.hpp"
#include "../../data_structs/Board.hpp"

#include "tables/BishopMap.hpp"
#include "../../data_structs/Move.hpp"
#include "tables/RookMap.hpp"
#include "tables/KnightMap.hpp"
#include "tables/KingMap.hpp"
#include "tables/WhitePawnMap.hpp"
#include "tables/BlackPawnMap.hpp"

#include <cassert>
#include <cinttypes>

struct ChessMechanics {
    // ------------------------------
    // Class inner types
    // ------------------------------

    enum checkType {
        slidingFigCheck,
        simpleFigCheck
    };

    enum class PinnedFigGen {
        WAllowedTiles,
        WoutAllowedTiles
    };

    // ------------------------------
    // Class Creation
    // ------------------------------

    ChessMechanics() = delete;

    explicit ChessMechanics(const Board &bd) : _board(bd) {}

    ChessMechanics(ChessMechanics &other) = delete;

    ChessMechanics &operator=(ChessMechanics &) = delete;

    ~ChessMechanics() = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] bool IsCheck() const {
        const int enemyCol = SwapColor(_board.MovingColor);
        const int kingsMsb = _board.GetKingMsbPos(_board.MovingColor);
        const uint64_t fullBoard = GetFullBitMap();

        // Checking rook's perspective
        const uint64_t enemyRooks = _board.GetFigBoard(enemyCol, rooksIndex);
        const uint64_t enemyQueens = _board.GetFigBoard(enemyCol, queensIndex);

        const uint64_t kingsRookPerspective = RookMap::GetMoves(kingsMsb, fullBoard);

        if ((kingsRookPerspective & (enemyRooks | enemyQueens)) != 0)
            return true;

        // Checking bishop's perspective
        const uint64_t enemyBishops = _board.GetFigBoard(enemyCol, bishopsIndex);

        const uint64_t kingsBishopPerspective = BishopMap::GetMoves(kingsMsb, fullBoard);

        if ((kingsBishopPerspective & (enemyBishops | enemyQueens)) != 0)
            return true;

        // checking knights attacks
        const uint64_t enemyKnights = _board.GetFigBoard(enemyCol, knightsIndex);

        const uint64_t knightsPerspective = KnightMap::GetMoves(kingsMsb);

        if ((knightsPerspective & (enemyKnights)) != 0)
            return true;

        // pawns checks
        const uint64_t enemyPawns = _board.GetFigBoard(enemyCol, pawnsIndex);
        const uint64_t pawnAttacks =
                enemyCol == WHITE ? WhitePawnMap::GetAttackFields(enemyPawns) : BlackPawnMap::GetAttackFields(
                        enemyPawns);

        if ((pawnAttacks & (MaxMsbPossible >> kingsMsb)) != 0)
            return true;

        return false;
    }


    // Gets occupancy maps, which simply indicates whether some field is occupied or not. Does not distinguish colors.
    [[nodiscard]] inline uint64_t GetFullBitMap() const {
        uint64_t map = 0;
        for (const auto m: _board.BitBoards) map |= m;
        return map;
    }

    // Gets occupancy maps, which simply indicates whether some field is occupied or not, by desired color figures.
    [[nodiscard]] inline uint64_t GetColBitMap(const int col) const {
        assert(col == 1 || col == 0);

        uint64_t map = 0;
        for (size_t i = 0; i < Board::BitBoardsPerCol; ++i) map |= _board.BitBoards[Board::BitBoardsPerCol * col + i];
        return map;
    }

    // does not check kings BitBoards!!!
    [[nodiscard]] inline size_t GetIndexOfContainingBitBoard(const uint64_t map, const int col) const {
        const size_t colIndex = col * Board::BitBoardsPerCol;
        size_t rv = 0;
        for (size_t i = 0; i < Board::BitBoardsPerCol; ++i) {
            rv += ((_board.BitBoards[colIndex + i] & map) != 0) * i;
        }
        return colIndex + rv;
    }

    /*      IMPORTANT NOTES:
    *  BlockedFieldsMap - indicates whether some field could be attacked by enemy figures in their next round.
    *  During generation process there are check test performed with counting, what yields 3 code branches inside main
    *  generation code. Map is mainly used when king is moving, allowing to simply predict whether king should
    *  move to that tile or not.
    *
    */

    // [blockedFigMap, checksCount, checkType]
    [[nodiscard]] std::tuple<uint64_t, uint8_t, uint8_t> GetBlockedFieldBitMap(uint64_t fullMap) const {
        assert(fullMap != 0 && "Full map is empty!");

        uint8_t checksCount{};
        uint8_t chT{};

        const int enemyCol = SwapColor(_board.MovingColor);
        const size_t enemyFigInd = enemyCol * Board::BitBoardsPerCol;
        const int allyKingShift = ConvertToReversedPos(_board.GetKingMsbPos(_board.MovingColor));
        const uint64_t allyKingMap = 1LLU << allyKingShift;

        // allows to also simply predict which tiles on the other side of the king are allowed.
        const uint64_t fullMapWoutKing = fullMap ^ allyKingMap;

        // King attacks generation.
        const uint64_t kingBlockedMap = KingMap::GetMoves(_board.GetKingMsbPos(enemyCol));

        // Rook attacks generation. Needs special treatment to correctly detect double check, especially with pawn promotion
        const auto [rookBlockedMap, checkCountRook] = _getRookBlockedMap(
                _board.BitBoards[enemyFigInd + rooksIndex] | _board.BitBoards[enemyFigInd + queensIndex],
                fullMapWoutKing,
                allyKingMap
        );

        // = 0, 1 or eventually 2 when promotion to rook like type happens
        checksCount += checkCountRook;

        // Bishop attacks generation.
        const uint64_t bishopBlockedMap = _blockIterativeGenerator(
                _board.BitBoards[enemyFigInd + bishopsIndex] | _board.BitBoards[enemyFigInd + queensIndex],
                [=](const int pos) {
                    return BishopMap::GetMoves(pos, fullMapWoutKing);
                }
        );

        // = 1 or 0 depending on whether hits or not
        const uint8_t wasCheckedByBishopFlag = (bishopBlockedMap & allyKingMap) >> allyKingShift;
        checksCount += wasCheckedByBishopFlag;

        // Pawns attacks generation.
        const uint64_t pawnsMap = _board.BitBoards[enemyFigInd + pawnsIndex];
        const uint64_t pawnBlockedMap =
                enemyCol == WHITE ? WhitePawnMap::GetAttackFields(pawnsMap) : BlackPawnMap::GetAttackFields(pawnsMap);

        // = 1 or 0 depending on whether hits or not
        const uint8_t wasCheckedByPawnFlag = (pawnBlockedMap & allyKingMap) >> allyKingShift;
        checksCount += wasCheckedByPawnFlag;

        // modifying check type
        chT += simpleFigCheck * wasCheckedByPawnFlag; // Note: king cannot be double-checked by simple figure

        // Knight attacks generation.
        const uint64_t knightBlockedMap = _blockIterativeGenerator(
                _board.BitBoards[enemyFigInd + knightsIndex],
                [=](const int pos) {
                    return KnightMap::GetMoves(pos);
                }
        );

        // = 1 or 0 depending on whether hits or not
        const uint8_t wasCheckedByKnightFlag = (knightBlockedMap & allyKingMap) >> allyKingShift;
        checksCount += wasCheckedByKnightFlag;

        // modifying check type
        chT += simpleFigCheck * wasCheckedByKnightFlag; // Note: king cannot be double-checked by simple figure

        const uint64_t blockedMap =
                kingBlockedMap | pawnBlockedMap | knightBlockedMap | rookBlockedMap | bishopBlockedMap;
        return {blockedMap, checksCount, chT};
    }

    [[nodiscard]] uint64_t GenerateAllowedTilesForPrecisedPinnedFig(uint64_t figBoard, uint64_t fullMap) const {
        assert(fullMap != 0 && "Full map is empty!");
        assert(CountOnesInBoard(figBoard) == 1 && "Only one figure should be pinned!");

        const int msbPos = ExtractMsbPos(figBoard);
        const uint64_t KingBoard = _board.BitBoards[_board.MovingColor * Board::BitBoardsPerCol + kingIndex];

        const uint64_t RookPerspectiveMoves = RookMap::GetMoves(msbPos, fullMap);
        if ((RookPerspectiveMoves & KingBoard) != 0)
            return RookPerspectiveMoves & RookMap::GetMoves(ExtractMsbPos(KingBoard), fullMap ^ figBoard);

        const uint64_t BishopPerspectiveMoves = BishopMap::GetMoves(msbPos, fullMap);
        return BishopPerspectiveMoves & BishopMap::GetMoves(ExtractMsbPos(KingBoard), fullMap ^ figBoard);
    }

    // returns [ pinnedFigMap, allowedTilesMap ]
    template<PinnedFigGen genType>
    [[nodiscard]] std::pair<uint64_t, uint64_t> GetPinnedFigsMap(int col, uint64_t fullMap) const {
        assert(fullMap != 0 && "Full map is empty!");
        assert(col == 1 || col == 0 && "Invalid color!");

        const size_t enemyCord = SwapColor(col) * Board::BitBoardsPerCol;

        const auto [pinnedByRooks, allowedRooks] = _getPinnedFigMaps<RookMap, genType>(
                fullMap, _board.BitBoards[enemyCord + rooksIndex] | _board.BitBoards[enemyCord + queensIndex]
        );

        const auto [pinnedByBishops, allowedBishops] = _getPinnedFigMaps<BishopMap, genType>(
                fullMap, _board.BitBoards[enemyCord + bishopsIndex] | _board.BitBoards[enemyCord + queensIndex]
        );

        return {pinnedByBishops | pinnedByRooks, allowedBishops | allowedRooks};
    }

    [[nodiscard]] uint64_t GetAllowedTilesWhenCheckedByNonSliding() const {
        uint64_t allowedTiles{};

        allowedTiles |= KingMap::GetSimpleFigCheckKnightsAllowedTiles(_board);
        allowedTiles |= KingMap::GetSimpleFigCheckPawnAllowedTiles(_board);

        return allowedTiles;
    }


    // ------------------------------
    // private methods
    // ------------------------------

private:


    /*  Function collects information used inside the SEE algorithm it contains:
     *  - attackersBitBoard - contains every type of figure that in current state of the board could attack given field
     *  - fullMap - contains every figure on the board
     *  - xrayMap - contains every figure that attack could be potentially unlocked after other figures move,
     *              that is: queens, bishops, rooks and pawns
     * */

    static std::pair<uint64_t, uint8_t>
    _getRookBlockedMap(uint64_t rookMap, uint64_t fullMapWoutKing, uint64_t kingMap) {
        assert(kingMap != 0 && "King map is empty!");

        uint64_t blockedTiles{};
        uint8_t checks{};

        while (rookMap) {
            const int msbPos = ExtractMsbPos(rookMap);

            const uint64_t moves = RookMap::GetMoves(msbPos, fullMapWoutKing);
            blockedTiles |= moves;
            checks += ((moves & kingMap) != 0);

            rookMap ^= MaxMsbPossible >> msbPos;
        }

        return {blockedTiles, checks};
    }

    template<class MoveGeneratorT>
    [[nodiscard]] inline static uint64_t _blockIterativeGenerator(uint64_t board, MoveGeneratorT mGen) {
        uint64_t blockedMap = 0;

        while (board != 0) {
            const int figPos = ExtractMsbPos(board);
            board ^= (MaxMsbPossible >> figPos);

            blockedMap |= mGen(figPos);
        }

        return blockedMap;
    }

    // returns [ pinnedFigMap, allowedTilesMap ]
    template<class MoveMapT, PinnedFigGen type>
    [[nodiscard]] std::pair<uint64_t, uint64_t>
    _getPinnedFigMaps(uint64_t fullMap, uint64_t possiblePinningFigs) const {
        uint64_t allowedTilesFigMap{};
        [[maybe_unused]] uint64_t pinnedFigMap{};

        const int kingPos = _board.GetKingMsbPos(_board.MovingColor);
        // generating figs seen from king's rook perspective
        const uint64_t kingFigPerspectiveAttackedFields = MoveMapT::GetMoves(kingPos, fullMap);
        const uint64_t kingFigPerspectiveAttackedFigs = kingFigPerspectiveAttackedFields & fullMap;

        // this functions should be called only in case of single check so the value below can only be either null or the
        // map of checking figure
        if constexpr (type == PinnedFigGen::WAllowedTiles)
            if (const uint64_t kingSeenEnemyFigs = kingFigPerspectiveAttackedFigs & possiblePinningFigs;
                    kingSeenEnemyFigs != 0) {
                const int msbPos = ExtractMsbPos(kingSeenEnemyFigs);
                const uint64_t moves = MoveMapT::GetMoves(msbPos, fullMap);

                allowedTilesFigMap = (moves & kingFigPerspectiveAttackedFields) | kingSeenEnemyFigs;
            }

        // removing figs seen by king
        const uint64_t cleanedMap = fullMap ^ kingFigPerspectiveAttackedFigs;

        // generating figs, which stayed behind first ones and are actually pinning ones
        const uint64_t kingSecondRookPerspective = MoveMapT::GetMoves(kingPos, cleanedMap);
        uint64_t pinningFigs = possiblePinningFigs & kingSecondRookPerspective;

        // generating fields which are both seen by king and pinning figure = field on which pinned figure stays
        while (pinningFigs != 0) {
            const int msbPos = ExtractMsbPos(pinningFigs);
            pinnedFigMap |= MoveMapT::GetMoves(msbPos, fullMap) & kingFigPerspectiveAttackedFigs;
            pinningFigs ^= MaxMsbPossible >> msbPos;
        }

        return {pinnedFigMap, allowedTilesFigMap};
    }

    // ------------------------------
    // Class fields
    // ------------------------------

protected:
    const Board &_board;
};


#endif // CHESSMECHANICS_H
