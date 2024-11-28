//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef MOVEGENERATOR_CUH
#define MOVEGENERATOR_CUH


#include "ChessMechanics.cuh"
#include "cuda_BitOperations.cuh"
#include "Move.cuh"
#include "Stack.cuh"

#include "BishopMap.cuh"
#include "BlackPawnMap.cuh"
#include "WhitePawnMap.cuh"
#include "KingMap.cuh"
#include "KnightMap.cuh"
#include "QueenMap.cuh"
#include "RookMap.cuh"
#include "cuda_Array.cuh"
#include "RookMapRuntime.cuh"
#include "BishopMapRuntime.cuh"

#include <cassert>
#include <type_traits>

#define GET_PAWN_FIELD(param) \
    __uint64_t param{}; \
    \
    if constexpr (std::is_same<MapT, WhitePawnMap>::value) { \
        param = WhitePawnMapConstants::param; \
    } else if constexpr (std::is_same<MapT, BlackPawnMap>::value) { \
        param = BlackPawnMapConstants::param; \
    } else { \
        assert(false && "Invalid pawn map type detected!"); \
    }                           \

__device__ static constexpr uint16_t PromoFlags[]{
        0, KnightFlag, BishopFlag, RookFlag, QueenFlag,
};

class MoveGenerator : ChessMechanics {

    enum MoveGenFlags : __uint32_t {
        EMPTY_FLAGS = 0,
        CHECK_CASTLINGS = 1,
        PROMOTE_PAWNS = 2,
        SELECT_FIGURES = 4,
        ASSUME_CHECK = 8,
        CONSIDER_EL_PASSANT = 16,
    };

    FAST_DCALL_ALWAYS static constexpr __uint32_t ExtractFlag(__uint32_t flags, MoveGenFlags flag) {
        return flags & flag;
    }

    FAST_DCALL_ALWAYS static constexpr bool IsFlagOn(__uint32_t flags, MoveGenFlags flag) {
        return ExtractFlag(flags, flag) != 0;
    }

public:
    Stack<cuda_Move>& stack;
    using payload = Stack<cuda_Move>::StackPayload;

    // ------------------------------
    // Class Creation
    // ------------------------------

    MoveGenerator() = delete;

    FAST_DCALL explicit MoveGenerator(const cuda_Board &bd, Stack<cuda_Move>& s) : ChessMechanics(bd), stack(s) {}

    MoveGenerator(MoveGenerator &other) = delete;

    MoveGenerator &operator=(MoveGenerator &) = delete;

    ~MoveGenerator() = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL payload GetMovesFast() {
        // Prepare crucial components and additionally detect whether we are at check and which figure type attacks king
        const __uint64_t fullMap = GetFullBitMap();
        const auto [blockedFigMap, checksCount, wasCheckedBySimple] = GetBlockedFieldBitMap(fullMap);

        assert(blockedFigMap != 0 && "Blocked fig map must at least contains fields controlled by king!");
        assert(
                checksCount <= 2 &&
                "We consider only 3 states: no-check, single-check, double-check -> invalid result was returned!"
        );

        payload results = stack.GetPayload();

        // depending on amount of checks branch to desired reaction
        switch (checksCount) {
            case 0:
                _upTo1Check(results, fullMap, blockedFigMap, EMPTY_FLAGS);
                break;
            case 1:
                _upTo1Check(results, fullMap, blockedFigMap, ASSUME_CHECK, wasCheckedBySimple);
                break;
            case 2:
                _doubleCheckGen(results, blockedFigMap);
                break;
            default:
                assert(false && "Invalid check count detected!");
                break;
        }

        return results;
    }

    [[nodiscard]] __device__ __uint64_t CountMoves(int depth) {
        cuda_Board bd = _board;
        return CountMovesRecursive(bd, depth);
    }

    [[nodiscard]] __device__ __uint64_t CountMovesRecursive(cuda_Board &bd, int depth) {
        if (depth == 0)
            return 1;

        MoveGenerator mGen{bd, stack};
        const auto moves = mGen.GetMovesFast();

        if (depth == 1) {
            stack.PopAggregate(moves);
            return moves.size;
        }

        __uint64_t sum{};

        VolatileBoardData data{bd};
        for (__uint32_t i = 0; i < moves.size; ++i) {
            cuda_Move::MakeMove(moves[i], bd);
            sum += CountMovesRecursive(bd, depth - 1);
            cuda_Move::UnmakeMove(moves[i], bd, data);
        }

        stack.PopAggregate(moves);
        return sum;
    }

    using ChessMechanics::IsCheck;

    // ------------------------------
    // private methods
    // ------------------------------

private:

    FAST_DCALL void _upTo1Check(payload &results, __uint64_t fullMap, __uint64_t blockedFigMap, __uint32_t flags,
                                bool wasCheckedBySimpleFig = false) {
        assert(fullMap != 0 && "Full map is empty!");
        static constexpr __uint64_t UNUSED = 0;

        const auto [pinnedFigsMap, allowedTilesMap] = [&]() -> thrust::pair<__uint64_t, __uint64_t> {
            if (!IsFlagOn(flags, ASSUME_CHECK)) {
                return GetPinnedFigsMap<ChessMechanics::PinnedFigGen::WoutAllowedTiles>(_board.MovingColor, fullMap);
            }

            if (!wasCheckedBySimpleFig) {
                return GetPinnedFigsMap<ChessMechanics::PinnedFigGen::WAllowedTiles>(_board.MovingColor, fullMap);
            }

            [[maybe_unused]] const auto [rv, _] =
                    GetPinnedFigsMap<ChessMechanics::PinnedFigGen::WoutAllowedTiles>(_board.MovingColor, fullMap);
            return {rv, GetAllowedTilesWhenCheckedByNonSliding()};
        }();

        // helping variable preparation
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_board.MovingColor));
        const __uint64_t allyMap = GetColBitMap(_board.MovingColor);

        // Specific figure processing
        _processFigMoves<KnightMap>(
                results, enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
        );

        _processFigMoves<BishopMap>(
                results, enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
        );

        _processFigMoves<RookMap>(
                results, enemyMap, allyMap, pinnedFigsMap,
                (IsFlagOn(flags, ASSUME_CHECK) ? ASSUME_CHECK : CHECK_CASTLINGS),
                UNUSED, allowedTilesMap
        );

        _processFigMoves<QueenMap>(
                results, enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
        );

        if (_board.MovingColor == WHITE) {
            _processPawnMoves<WhitePawnMap>(
                    results, enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), allowedTilesMap
            );
        } else {
            _processPawnMoves<BlackPawnMap>(
                    results, enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), allowedTilesMap
            );
        }

        _processPlainKingMoves(results, blockedFigMap, allyMap, enemyMap);

        if (!IsFlagOn(flags, ASSUME_CHECK)) {
            _processKingCastlings(results, blockedFigMap, fullMap);
        }
    }

    FAST_DCALL void _doubleCheckGen(payload &results, __uint64_t blockedFigMap) {
        const __uint64_t allyMap = GetColBitMap(_board.MovingColor);
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_board.MovingColor));
        _processPlainKingMoves(results, blockedFigMap, allyMap, enemyMap);
    }

    template<class MapT>
    FAST_DCALL void _processPawnMoves(
            payload &results, __uint64_t enemyMap, __uint64_t allyMap, __uint64_t pinnedFigMap, __uint32_t flags,
            __uint64_t allowedMoveFilter = 0
    ) {
        GET_PAWN_FIELD(PromotingMask);

        const __uint64_t promotingPawns = _board.BitBoards[MapT::GetBoardIndex(0)] & PromotingMask;
        const __uint64_t nonPromotingPawns = _board.BitBoards[MapT::GetBoardIndex(0)] ^ promotingPawns;

        _processFigMoves<MapT>(
                results, enemyMap, allyMap, pinnedFigMap,
                CONSIDER_EL_PASSANT | SELECT_FIGURES | ExtractFlag(flags, ASSUME_CHECK),
                nonPromotingPawns, allowedMoveFilter
        );

        // During quiesce search we should also check all promotions so GenOnlyTacticalMoves is false
        if (promotingPawns) {
            _processFigMoves<MapT>(
                    results, enemyMap, allyMap, pinnedFigMap,
                    SELECT_FIGURES | ExtractFlag(flags, ASSUME_CHECK) | PROMOTE_PAWNS,
                    promotingPawns,
                    allowedMoveFilter
            );
        }

        _processElPassantMoves<MapT>(
                results, allyMap | enemyMap, pinnedFigMap, IsFlagOn(flags, ASSUME_CHECK), allowedMoveFilter
        );
    }

    // TODO: Consider different solution?
    template<class MapT>
    FAST_DCALL void _processElPassantMoves(
            payload &results, __uint64_t fullMap, __uint64_t pinnedFigMap, bool isCheck,
            __uint64_t allowedMoveFilter = 0
    ) {
        assert(fullMap != 0 && "Full map is empty!");

        if (_board.ElPassantField == InvalidElPassantBitBoard) {
            return;
        }

        // calculation preparation
        const __uint64_t suspectedFields = MapT::GetElPassantSuspectedFields(_board.ElPassantField);
        const __uint32_t enemyCord = SwapColor(_board.MovingColor) * BitBoardsPerCol;
        const __uint64_t enemyRookFigs = _board.BitBoards[enemyCord + queensIndex] | _board.BitBoards[enemyCord + rooksIndex];
        __uint64_t possiblePawnsToMove = _board.BitBoards[MapT::GetBoardIndex(0)] & suspectedFields;

        GET_PAWN_FIELD(EnemyElPassantMask);

        while (possiblePawnsToMove) {
            const __uint64_t pawnMap = cuda_MaxMsbPossible >> ExtractMsbPos(possiblePawnsToMove);

            // checking whether move would affect horizontal line attacks on king
            const __uint64_t processedPawns = pawnMap | _board.ElPassantField;
            const __uint64_t cleanedFromPawnsMap = fullMap ^ processedPawns;
            if (const __uint64_t kingHorizontalLine =
                        RookMap::GetMoves(_board.GetKingMsbPos(_board.MovingColor), cleanedFromPawnsMap) & EnemyElPassantMask;
                    (kingHorizontalLine & enemyRookFigs) != 0) {
                return;
            }

            const __uint64_t moveMap = MapT::GetElPassantMoveField(_board.ElPassantField);

            // checking whether moving some pawns would undercover king on some line
            if ((processedPawns & pinnedFigMap) != 0) {
                // two separate situations that's need to be considered, every pawn that participate in el passant move
                // should be unblocked on specific lines

                if ((pawnMap & pinnedFigMap) != 0 &&
                    (GenerateAllowedTilesForPrecisedPinnedFig(pawnMap, fullMap) & moveMap) == 0) {
                    possiblePawnsToMove ^= pawnMap;
                    continue;
                }
                if ((GenerateAllowedTilesForPrecisedPinnedFig(_board.ElPassantField, fullMap) & moveMap) == 0) {
                    possiblePawnsToMove ^= pawnMap;
                    continue;
                }
            }

            // When king is checked only if move is going to allow tile el passant is correct
            if (isCheck && (moveMap & allowedMoveFilter) == 0 && (_board.ElPassantField & allowedMoveFilter) == 0) {
                possiblePawnsToMove ^= pawnMap;
                continue;
            }

            // preparing basic move information
            cuda_Move mv{};
            mv.SetStartField(ExtractMsbPos(pawnMap));
            mv.SetStartBoardIndex(MapT::GetBoardIndex(0));
            mv.SetTargetField(ExtractMsbPos(moveMap));
            mv.SetTargetBoardIndex(MapT::GetBoardIndex(0));
            mv.SetKilledBoardIndex(MapT::GetEnemyPawnBoardIndex());
            mv.SetKilledFigureField(ExtractMsbPos(_board.ElPassantField));
            mv.SetElPassantField(InvalidElPassantField);
            mv.SetCasltingRights(_board.Castlings);
            mv.SetMoveType(CaptureFlag);

            results.Push(stack, mv);
            possiblePawnsToMove ^= pawnMap;
        }
    }

    // TODO: Compare with simple if searching loop
    // TODO: propagate checkForCastling?
    template<class MapT>
    __device__ void _processFigMoves(
            payload &results, __uint64_t enemyMap, __uint64_t allyMap, __uint64_t pinnedFigMap, __uint32_t flags,
            __uint64_t figureSelector = 0, __uint64_t allowedMoveSelector = 0
    )  {
        assert(enemyMap != 0 && "Enemy map is empty!");
        assert(allyMap != 0 && "Ally map is empty!");

        const __uint64_t fullMap = enemyMap | allyMap;
        __uint64_t pinnedOnes = pinnedFigMap & _board.BitBoards[MapT::GetBoardIndex(_board.MovingColor)];
        __uint64_t unpinnedOnes = _board.BitBoards[MapT::GetBoardIndex(_board.MovingColor)] ^ pinnedOnes;

        // applying filter if needed
        if (IsFlagOn(flags, SELECT_FIGURES)) {
            pinnedOnes &= figureSelector;
            unpinnedOnes &= figureSelector;
        }

        // saving results of previous el passant field, used only when the figure is not a pawn

        // processing unpinned moves
        while (unpinnedOnes) {
            // processing moves
            const int figPos = ExtractMsbPos(unpinnedOnes);
            const __uint64_t figBoard = cuda_MaxMsbPossible >> figPos;

            // selecting allowed moves if in check
            const __uint64_t figMoves = [&]() constexpr {
                if (!IsFlagOn(flags, ASSUME_CHECK)) {
                    return MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap;
                }
                else {
                    return MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap & allowedMoveSelector;
                }
            }();

            // Performing checks for castlings
            __uint32_t updatedCastlings = _board.Castlings;
            if (IsFlagOn(flags, CHECK_CASTLINGS)) {
                SetBitBoardBit(updatedCastlings, RookMap::GetMatchingCastlingIndex(_board, figBoard), false);
            }

            // preparing moves
            const __uint64_t attackMoves = figMoves & enemyMap;
            [[maybe_unused]] const __uint64_t nonAttackingMoves = figMoves ^ attackMoves;

            // processing move consequences
            _processNonAttackingMoves(
                    results, nonAttackingMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard,
                    updatedCastlings, ExtractFlag(flags, CONSIDER_EL_PASSANT) | ExtractFlag(flags, PROMOTE_PAWNS)
            );

            _processAttackingMoves(
                    results, attackMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard, updatedCastlings,
                    IsFlagOn(flags, PROMOTE_PAWNS)
            );

            unpinnedOnes ^= figBoard;
        }

        // if a check is detected, the pinned figure stays in place
        if (IsFlagOn(flags, ASSUME_CHECK))
            return;

        // processing pinned moves
        // Note: corner Rook possibly applicable to castling cannot be pinned
        while (pinnedOnes) {
            // processing moves
            const int figPos = ExtractMsbPos(pinnedOnes);
            const __uint64_t figBoard = cuda_MaxMsbPossible >> figPos;
            const __uint64_t allowedTiles = GenerateAllowedTilesForPrecisedPinnedFig(figBoard, fullMap);
            const __uint64_t figMoves = MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap & allowedTiles;
            // TODO: check applied here?
            // TODO: breaking if there?

            // preparing moves
            const __uint64_t attackMoves = figMoves & enemyMap;
            [[maybe_unused]] const __uint64_t nonAttackingMoves = figMoves ^ attackMoves;

            // processing move consequences

            _processNonAttackingMoves(
                    results, nonAttackingMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard,
                    _board.Castlings, ExtractFlag(flags, CONSIDER_EL_PASSANT) | ExtractFlag(flags, PROMOTE_PAWNS)
            );

            // TODO: There is exactly one move possible
            _processAttackingMoves(
                    results, attackMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard, _board.Castlings,
                    IsFlagOn(flags, PROMOTE_PAWNS)
            );

            pinnedOnes ^= figBoard;
        }
    }

    __device__ void _processNonAttackingMoves(
            payload &results, __uint64_t nonAttackingMoves, __uint32_t figBoardIndex, __uint64_t startField,
            __uint32_t castlings, __uint32_t flags
    ) {
        assert(figBoardIndex < BitBoardsCount && "Invalid figure cuda_Board index!");

        while (nonAttackingMoves) {
            // extracting moves
            const int movePos = ExtractMsbPos(nonAttackingMoves);
            const __uint64_t moveBoard = cuda_MaxMsbPossible >> movePos;

            cuda_Move mv{};

            mv.SetStartField(ExtractMsbPos(startField));
            mv.SetStartBoardIndex(figBoardIndex);
            mv.SetTargetField(movePos);
            mv.SetCasltingRights(castlings);
            mv.SetKilledBoardIndex(SentinelBoardIndex);

            if (IsFlagOn(flags, CONSIDER_EL_PASSANT)) {
                const auto result = (_board.MovingColor == WHITE ?
                                     WhitePawnMap::GetElPassantField(moveBoard, startField) :
                                     BlackPawnMap::GetElPassantField(moveBoard, startField)
                );

                mv.SetElPassantField(result == 0 ? InvalidElPassantField : ExtractMsbPos(result));
            } else {
                mv.SetElPassantField(InvalidElPassantField);
            }

            mv.SetTargetBoardIndex(IsFlagOn(flags, PROMOTE_PAWNS) ? _board.MovingColor * BitBoardsPerCol + queensIndex
                                                                  : figBoardIndex);
            mv.SetMoveType(IsFlagOn(flags, PROMOTE_PAWNS) ? PromoFlag | PromoFlags[queensIndex] : 0);

            results.Push(stack, mv);
            nonAttackingMoves ^= moveBoard;
        }
    }
    FAST_DCALL void _processAttackingMoves(
            payload &results, __uint64_t attackingMoves, __uint32_t figBoardIndex, __uint64_t startField,
            __uint32_t castlings, bool promotePawns
    ) {
        assert(figBoardIndex < BitBoardsCount && "Invalid figure cuda_Board index!");

        while (attackingMoves) {
            // extracting moves
            const int movePos = ExtractMsbPos(attackingMoves);
            const __uint64_t moveBoard = cuda_MaxMsbPossible >> movePos;
            const __uint32_t attackedFigBoardIndex = GetIndexOfContainingBitBoard(moveBoard,
                                                                                  SwapColor(_board.MovingColor));

            cuda_Move mv{};

            mv.SetStartField(ExtractMsbPos(startField));
            mv.SetStartBoardIndex(figBoardIndex);
            mv.SetTargetField(movePos);
            mv.SetKilledBoardIndex(attackedFigBoardIndex);
            mv.SetKilledFigureField(movePos);
            mv.SetElPassantField(InvalidElPassantField);
            mv.SetCasltingRights(castlings);
            mv.SetMoveType(CaptureFlag);
            mv.SetTargetBoardIndex(promotePawns ? _board.MovingColor * BitBoardsPerCol + queensIndex : figBoardIndex);
            mv.SetMoveType(promotePawns ? PromoFlag | PromoFlags[queensIndex] : 0);

            results.Push(stack, mv);
            attackingMoves ^= moveBoard;
        }
    }

    __device__ void
    _processPlainKingMoves(payload &results, __uint64_t blockedFigMap, __uint64_t allyMap, __uint64_t enemyMap) {
        assert(allyMap != 0 && "Ally map is empty!");
        assert(enemyMap != 0 && "Enemy map is empty!");

        // generating moves
        const __uint64_t kingMoves = KingMap::GetMoves(_board.GetKingMsbPos(_board.MovingColor)) &
                ~blockedFigMap & ~allyMap;

        __uint64_t attackingMoves = kingMoves & enemyMap;
        [[maybe_unused]] __uint64_t nonAttackingMoves = kingMoves ^ attackingMoves;

        // preparing variables
        auto castlings = _board.Castlings;
        SetBitBoardBit(castlings, _board.MovingColor * CastlingsPerColor + KingCastlingIndex, false);
        SetBitBoardBit(castlings, _board.MovingColor * CastlingsPerColor + QueenCastlingIndex, false);

        const int oldKingPos = ExtractMsbPos(_board.BitBoards[_board.MovingColor * BitBoardsPerCol + kingIndex]);

        // processing simple non-attacking moves
        while (nonAttackingMoves) {
            // extracting new king position data
            const int newPos = ExtractMsbPos(nonAttackingMoves);

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(oldKingPos);
            mv.SetStartBoardIndex(_board.MovingColor * BitBoardsPerCol + kingIndex);
            mv.SetTargetField(newPos);
            mv.SetTargetBoardIndex(_board.MovingColor * BitBoardsPerCol + kingIndex);
            mv.SetKilledBoardIndex(SentinelBoardIndex);
            mv.SetElPassantField(InvalidElPassantField);
            mv.SetCasltingRights(castlings);

            results.Push(stack, mv);

            nonAttackingMoves ^= (cuda_MaxMsbPossible >> newPos);
        }

        // processing slightly more complicated attacking moves
        while (attackingMoves) {
            // extracting new king position data
            const int newPos = ExtractMsbPos(attackingMoves);
            const __uint64_t newKingBoard = cuda_MaxMsbPossible >> newPos;

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(oldKingPos);
            mv.SetStartBoardIndex(_board.MovingColor * BitBoardsPerCol + kingIndex);
            mv.SetTargetField(newPos);
            mv.SetTargetBoardIndex(_board.MovingColor * BitBoardsPerCol + kingIndex);
            mv.SetKilledBoardIndex(GetIndexOfContainingBitBoard(newKingBoard, SwapColor(_board.MovingColor)));
            mv.SetKilledFigureField(newPos);
            mv.SetElPassantField(InvalidElPassantField);
            mv.SetCasltingRights(castlings);
            mv.SetMoveType(CaptureFlag);

            results.Push(stack, mv);

            attackingMoves ^= newKingBoard;
        }
    }

    // TODO: simplify ifs??
    // TODO: cleanup left castling available when rook is dead then propagate no castling checking?
    FAST_DCALL void _processKingCastlings(payload &results, __uint64_t blockedFigMap, __uint64_t fullMap) {
        assert(fullMap != 0 && "Full map is empty!");

        for (__uint32_t i = 0; i < CastlingsPerColor; ++i) {
            if (const __uint32_t castlingIndex = _board.MovingColor * CastlingsPerColor + i;
                    _board.GetCastlingRight(castlingIndex) &&
                    (CastlingsRookMaps[castlingIndex] &
                     _board.BitBoards[_board.MovingColor * BitBoardsPerCol + rooksIndex]) != 0 &&
                    (CastlingSensitiveFields[castlingIndex] & blockedFigMap) == 0 &&
                    (CastlingTouchedFields[castlingIndex] & fullMap) == 0) {

                auto castlings = _board.Castlings;
                SetBitBoardBit(castlings, _board.MovingColor * CastlingsPerColor + KingCastlingIndex, false);
                SetBitBoardBit(castlings, _board.MovingColor * CastlingsPerColor + QueenCastlingIndex, false);

                cuda_Move mv{};

                // preparing basic move info
                mv.SetStartField(ExtractMsbPos(DefaultKingBoards[_board.MovingColor]));
                mv.SetStartBoardIndex(_board.MovingColor * BitBoardsPerCol + kingIndex);
                mv.SetTargetField(CastlingNewKingPos[castlingIndex]);
                mv.SetTargetBoardIndex(_board.MovingColor * BitBoardsPerCol + kingIndex);
                mv.SetKilledBoardIndex(_board.MovingColor * BitBoardsPerCol + rooksIndex);
                mv.SetKilledFigureField(ExtractMsbPos(CastlingsRookMaps[castlingIndex]));
                mv.SetElPassantField(InvalidElPassantField);
                mv.SetCasltingRights(castlings);
                mv.SetCastlingType(1 + castlingIndex); // God only knows why I made a sentinel at index 0 at this moment
                mv.SetMoveType(CastlingFlag);

                results.Push(stack, mv);
            }
        }
    }
};

#endif // MOVEGENERATOR_CUH
