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

__device__ static constexpr __uint64_t CASTLING_PSEUDO_LEGAL_BLOCKED = 0;

struct MoveGenerator : ChessMechanics {
    Stack<cuda_Move>& stack;
    using payload = Stack<cuda_Move>::StackPayload;

    // ------------------------------
    // Class Creation
    // ------------------------------

    MoveGenerator() = delete;

    FAST_CALL explicit MoveGenerator(const cuda_Board &bd, Stack<cuda_Move>& s) : ChessMechanics(bd), stack(s) {}

    MoveGenerator(MoveGenerator &other) = delete;

    MoveGenerator &operator=(MoveGenerator &) = delete;

    ~MoveGenerator() = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    template<bool GenOnlyTacticalMoves = false>
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
                _noCheckGen<GenOnlyTacticalMoves>(results, fullMap, blockedFigMap);
                break;
            case 1:
                _singleCheckGen<GenOnlyTacticalMoves>(results, fullMap, blockedFigMap, wasCheckedBySimple);
                break;
            case 2:
                _doubleCheckGen<GenOnlyTacticalMoves>(results, blockedFigMap);
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
        for (size_t i = 0; i < moves.size; ++i) {
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

    template<class MapT>
    [[nodiscard]] FAST_CALL bool _isGivingCheck(const int msbPos, const __uint64_t fullMap, const int enemyColor) const {
        const __uint64_t enemyKing = _board.BitBoards[enemyColor * BitBoardsPerCol + kingIndex];
        const __uint64_t moves = MapT::GetMoves(msbPos, fullMap, 0);

        return (enemyKing & moves) != 0;
    }

    template<class MapT>
    [[nodiscard]] FAST_CALL bool _isPawnGivingCheck(const __uint64_t pawnBitMap) const {
        const int enemyColor = SwapColor(MapT::GetColor());
        const __uint64_t enemyKing = _board.BitBoards[enemyColor * BitBoardsPerCol + kingIndex];
        const __uint64_t moves = MapT::GetAttackFields(pawnBitMap);

        return (enemyKing & moves) != 0;
    }

    template<class MapT>
    [[nodiscard]] FAST_CALL bool
    _isPromotingPawnGivingCheck(const int msbPos, const __uint64_t fullMap, const int targetBitBoardIndex) const {
        static constexpr __uint64_t (*moveGenerators[])(__uint32_t, __uint64_t, __uint64_t){
                nullptr, KnightMap::GetMoves, BishopMap::GetMoves, RookMap::GetMoves, QueenMap::GetMoves,
        };

        const int color = MapT::GetColor();
        const int enemyColor = SwapColor(color);
        const __uint64_t enemyKing = _board.BitBoards[enemyColor * BitBoardsPerCol + kingIndex];
        auto func = moveGenerators[targetBitBoardIndex - color * BitBoardsPerCol];
        const __uint64_t moves = func(msbPos, fullMap, 0);

        return (enemyKing & moves) != 0;
    }

    template<bool GenOnlyTacticalMoves>
    FAST_CALL void _noCheckGen(payload &results, __uint64_t fullMap, __uint64_t blockedFigMap) {
        assert(fullMap != 0 && "Full map is empty!");

        [[maybe_unused]] const auto [pinnedFigsMap, _] =
                GetPinnedFigsMap<ChessMechanics::PinnedFigGen::WoutAllowedTiles>(_board.MovingColor, fullMap);

        const __uint64_t enemyMap = GetColBitMap(SwapColor(_board.MovingColor));
        const __uint64_t allyMap = GetColBitMap(_board.MovingColor);

        _processFigMoves<GenOnlyTacticalMoves, KnightMap>(
                results, enemyMap, allyMap, pinnedFigsMap
        );

        _processFigMoves<GenOnlyTacticalMoves, BishopMap>(
                results, enemyMap, allyMap, pinnedFigsMap
        );

        _processFigMoves<GenOnlyTacticalMoves, RookMap, true>(
                results, enemyMap, allyMap, pinnedFigsMap
        );

        _processFigMoves<GenOnlyTacticalMoves, QueenMap>(
                results, enemyMap, allyMap, pinnedFigsMap
        );

        if (_board.MovingColor == WHITE)
            _processPawnMoves<GenOnlyTacticalMoves, WhitePawnMap>(
                    results, enemyMap, allyMap, pinnedFigsMap
            );
        else
            _processPawnMoves<GenOnlyTacticalMoves, BlackPawnMap>(
                    results, enemyMap, allyMap, pinnedFigsMap
            );

        _processPlainKingMoves<GenOnlyTacticalMoves>(results, blockedFigMap, allyMap, enemyMap);

        if constexpr (!GenOnlyTacticalMoves)
            _processKingCastlings(results, blockedFigMap, fullMap);
    }

    template<bool GenOnlyTacticalMoves>
    FAST_CALL void _singleCheckGen(payload &results, __uint64_t fullMap, __uint64_t blockedFigMap, bool wasCheckedBySimpleFig) {
        assert(fullMap != 0 && "Full map is empty!");

        static constexpr __uint64_t UNUSED = 0;

        // simplifying figure search by distinguishing check types
        const auto [pinnedFigsMap, allowedTilesMap] = [&]() -> std::pair<__uint64_t, __uint64_t> {
            if (!wasCheckedBySimpleFig)
                return GetPinnedFigsMap<ChessMechanics::PinnedFigGen::WAllowedTiles>(_board.MovingColor, fullMap);

            [[maybe_unused]] const auto [rv, _] =
                    GetPinnedFigsMap<ChessMechanics::PinnedFigGen::WoutAllowedTiles>(_board.MovingColor, fullMap);
            return {rv, GetAllowedTilesWhenCheckedByNonSliding()};
        }();

        // helping variable preparation
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_board.MovingColor));
        const __uint64_t allyMap = GetColBitMap(_board.MovingColor);

        // Specific figure processing
        _processFigMoves<GenOnlyTacticalMoves, KnightMap, false, false, false, true>(
                results, enemyMap, allyMap, pinnedFigsMap, UNUSED, allowedTilesMap
        );

        _processFigMoves<GenOnlyTacticalMoves, BishopMap, false, false, false, true>(
                results, enemyMap, allyMap, pinnedFigsMap, UNUSED, allowedTilesMap
        );

        _processFigMoves<GenOnlyTacticalMoves, RookMap, true, false, false, true>(
                results, enemyMap, allyMap, pinnedFigsMap, UNUSED, allowedTilesMap
        );

        _processFigMoves<GenOnlyTacticalMoves, QueenMap, false, false, false, true>(
                results, enemyMap, allyMap, pinnedFigsMap, UNUSED, allowedTilesMap
        );

        if (_board.MovingColor == WHITE)
            _processPawnMoves<GenOnlyTacticalMoves, WhitePawnMap, true>(
                    results, enemyMap, allyMap, pinnedFigsMap, allowedTilesMap
            );
        else
            _processPawnMoves<GenOnlyTacticalMoves, BlackPawnMap, true>(
                    results, enemyMap, allyMap, pinnedFigsMap, allowedTilesMap
            );

        _processPlainKingMoves<GenOnlyTacticalMoves>(results, blockedFigMap, allyMap, enemyMap);
    }

    template<bool GenOnlyTacticalMoves>
    FAST_CALL void _doubleCheckGen(payload &results, __uint64_t blockedFigMap) {
        const __uint64_t allyMap = GetColBitMap(_board.MovingColor);
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_board.MovingColor));
        _processPlainKingMoves<GenOnlyTacticalMoves>(results, blockedFigMap, allyMap, enemyMap);
    }

    template<bool GenOnlyTacticalMoves, class MapT, bool isCheck = false>
    FAST_CALL void _processPawnMoves(
            payload &results, __uint64_t enemyMap, __uint64_t allyMap, __uint64_t pinnedFigMap,
            [[maybe_unused]] __uint64_t allowedMoveFilter = 0
    ) {
        GET_PAWN_FIELD(PromotingMask);

        const __uint64_t promotingPawns = _board.BitBoards[MapT::GetBoardIndex(0)] & PromotingMask;
        const __uint64_t nonPromotingPawns = _board.BitBoards[MapT::GetBoardIndex(0)] ^ promotingPawns;

        _processFigMoves<
                GenOnlyTacticalMoves, MapT, false, false, true, isCheck, MapT::GetElPassantField>(
                results, enemyMap, allyMap, pinnedFigMap, nonPromotingPawns, allowedMoveFilter
        );

        // During quiesce search we should also check all promotions so GenOnlyTacticalMoves is false
        if (promotingPawns)
            _processFigMoves<false, MapT, false, true, true, isCheck>(
                    results, enemyMap, allyMap, pinnedFigMap, promotingPawns, allowedMoveFilter
            );

        _processElPassantMoves<MapT, isCheck>(
                results, allyMap | enemyMap, pinnedFigMap, allowedMoveFilter
        );
    }

    // TODO: Consider different solution?
    template<class MapT, bool isCheck = false>
    FAST_DCALL void _processElPassantMoves(
            payload &results, __uint64_t fullMap, __uint64_t pinnedFigMap, [[maybe_unused]] __uint64_t allowedMoveFilter = 0
    ) {
        assert(fullMap != 0 && "Full map is empty!");

        if (_board.ElPassantField == InvalidElPassantBitBoard)
            return;

        // calculation preparation
        const __uint64_t suspectedFields = MapT::GetElPassantSuspectedFields(_board.ElPassantField);
        const size_t enemyCord = SwapColor(_board.MovingColor) * BitBoardsPerCol;
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
                    (kingHorizontalLine & enemyRookFigs) != 0)
                return;

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
            if constexpr (isCheck)
                if ((moveMap & allowedMoveFilter) == 0 && (_board.ElPassantField & allowedMoveFilter) == 0) {
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

            if (_isPawnGivingCheck<MapT>(moveMap))
                mv.SetCheckType();


            results.Push(stack, mv);
            possiblePawnsToMove ^= pawnMap;
        }
    }

    // TODO: Compare with simple if searching loop
    // TODO: propagate checkForCastling?
    template<
            bool GenOnlyTacticalMoves, class MapT, bool checkForCastling = false,
            bool promotePawns = false, bool selectFigures = false, bool isCheck = false,
            __uint64_t (*elPassantFieldDeducer)(__uint64_t, __uint64_t) = nullptr>
    __device__ void _processFigMoves(
            payload &results, __uint64_t enemyMap, __uint64_t allyMap, __uint64_t pinnedFigMap,
            [[maybe_unused]] __uint64_t figureSelector = 0, [[maybe_unused]] __uint64_t allowedMoveSelector = 0
    )  {
        assert(enemyMap != 0 && "Enemy map is empty!");
        assert(allyMap != 0 && "Ally map is empty!");

        const __uint64_t fullMap = enemyMap | allyMap;
        __uint64_t pinnedOnes = pinnedFigMap & _board.BitBoards[MapT::GetBoardIndex(_board.MovingColor)];
        __uint64_t unpinnedOnes = _board.BitBoards[MapT::GetBoardIndex(_board.MovingColor)] ^ pinnedOnes;

        // applying filter if needed
        if constexpr (selectFigures) {
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
                if constexpr (!isCheck)
                    return MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap;
                if constexpr (isCheck)
                    return MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap & allowedMoveSelector;
            }();

            // Performing checks for castlings
            __uint32_t updatedCastlings = _board.Castlings;
            if constexpr (checkForCastling)
                SetBitBoardBit(updatedCastlings, RookMap::GetMatchingCastlingIndex(_board, figBoard), false);

            // preparing moves
            const __uint64_t attackMoves = figMoves & enemyMap;
            [[maybe_unused]] const __uint64_t nonAttackingMoves = figMoves ^ attackMoves;

            // processing move consequences

            if constexpr (!GenOnlyTacticalMoves)
                _processNonAttackingMoves<MapT, promotePawns, elPassantFieldDeducer>(
                        results, nonAttackingMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard,
                        updatedCastlings, fullMap
                );

            _processAttackingMoves<MapT, GenOnlyTacticalMoves, promotePawns>(
                    results, attackMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard, updatedCastlings,
                    fullMap
            );

            unpinnedOnes ^= figBoard;
        }

        // if a check is detected, the pinned figure stays in place
        if constexpr (isCheck)
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

            if constexpr (!GenOnlyTacticalMoves)
                _processNonAttackingMoves<MapT, promotePawns, elPassantFieldDeducer>(
                        results, nonAttackingMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard,
                        _board.Castlings, fullMap
                );

            // TODO: There is exactly one move possible
            _processAttackingMoves<MapT, GenOnlyTacticalMoves, promotePawns>(
                    results, attackMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard, _board.Castlings,
                    fullMap
            );

            pinnedOnes ^= figBoard;
        }
    }

    // TODO: improve readability of code below
    template<
            class MapT, bool promotePawns,
            __uint64_t (*elPassantFieldDeducer)(__uint64_t, __uint64_t) = nullptr>
    __device__ void _processNonAttackingMoves(
            payload &results, __uint64_t nonAttackingMoves, size_t figBoardIndex, __uint64_t startField,
            __uint32_t castlings, __uint64_t fullMap
    ) {
        assert(figBoardIndex < BitBoardsCount && "Invalid figure cuda_Board index!");

        while (nonAttackingMoves) {
            // extracting moves
            const int movePos = ExtractMsbPos(nonAttackingMoves);
            const __uint64_t moveBoard = cuda_MaxMsbPossible >> movePos;

            if constexpr (!promotePawns)
                // simple figure case
            {
                cuda_Move mv{};

                // preparing basic move info
                mv.SetStartField(ExtractMsbPos(startField));
                mv.SetStartBoardIndex(figBoardIndex);
                mv.SetTargetField(movePos);
                mv.SetTargetBoardIndex(figBoardIndex);
                mv.SetKilledBoardIndex(SentinelBoardIndex);

                if (figBoardIndex == wPawnsIndex && _isPawnGivingCheck<WhitePawnMap>(moveBoard))
                    mv.SetCheckType();
                else if (figBoardIndex == bPawnsIndex && _isPawnGivingCheck<BlackPawnMap>(moveBoard))
                    mv.SetCheckType();
                else if (_isGivingCheck<MapT>(movePos, fullMap ^ startField, SwapColor(figBoardIndex > wKingIndex)))
                    mv.SetCheckType();

                // if el passant line is passed when a figure moved to these line flags will turn on
                if constexpr (elPassantFieldDeducer != nullptr) {
                    // TODO: CHANGED TEMP
                    if (const auto result = elPassantFieldDeducer(moveBoard, startField); result == 0)
                        mv.SetElPassantField(InvalidElPassantField);
                    else
                        mv.SetElPassantField(ExtractMsbPos(result));
                } else
                    mv.SetElPassantField(InvalidElPassantField);
                mv.SetCasltingRights(castlings);

                results.Push(stack, mv);
            }
            if constexpr (promotePawns)
                // upgrading pawn case
            {
                static constexpr size_t startInd = queensIndex;
                static constexpr size_t limitInd = queensIndex - 1;

                // iterating through upgradable pieces
                for (size_t i = startInd; i > limitInd; --i) {
                    const auto targetBoard = _board.MovingColor * BitBoardsPerCol + i;

                    cuda_Move mv{};

                    // preparing basic move info
                    mv.SetStartField(ExtractMsbPos(startField));
                    mv.SetStartBoardIndex(figBoardIndex);
                    mv.SetTargetField(movePos);
                    mv.SetTargetBoardIndex(targetBoard);
                    mv.SetKilledBoardIndex(SentinelBoardIndex);
                    mv.SetElPassantField(InvalidElPassantField);
                    mv.SetCasltingRights(castlings);
                    mv.SetMoveType(PromoFlag | PromoFlags[i]);

                    if (_isPromotingPawnGivingCheck<MapT>(movePos, fullMap ^ startField, targetBoard))
                        mv.SetCheckType();

                    results.Push(stack, mv);
                }
            }

            nonAttackingMoves ^= moveBoard;
        }
    }

    template<class MapT, bool IsQsearch, bool promotePawns>
    __device__ void _processAttackingMoves(
            payload &results, __uint64_t attackingMoves, size_t figBoardIndex, __uint64_t startField,
            __uint32_t castlings, __uint64_t fullMap
    ) {
        assert(figBoardIndex < BitBoardsCount && "Invalid figure cuda_Board index!");

        while (attackingMoves) {
            // extracting moves
            const int movePos = ExtractMsbPos(attackingMoves);
            const __uint64_t moveBoard = cuda_MaxMsbPossible >> movePos;
            const size_t attackedFigBoardIndex = GetIndexOfContainingBitBoard(moveBoard, SwapColor(_board.MovingColor));

            if constexpr (!promotePawns)
                // simple figure case
            {
                cuda_Move mv{};

                // preparing basic move info
                mv.SetStartField(ExtractMsbPos(startField));
                mv.SetStartBoardIndex(figBoardIndex);
                mv.SetTargetField(movePos);
                mv.SetTargetBoardIndex(figBoardIndex);
                mv.SetKilledBoardIndex(attackedFigBoardIndex);
                mv.SetKilledFigureField(movePos);
                mv.SetElPassantField(InvalidElPassantField);
                mv.SetCasltingRights(castlings);
                mv.SetMoveType(CaptureFlag);

                if (figBoardIndex == wPawnsIndex && _isPawnGivingCheck<WhitePawnMap>(moveBoard))
                    mv.SetCheckType();
                else if (figBoardIndex == bPawnsIndex && _isPawnGivingCheck<BlackPawnMap>(moveBoard))
                    mv.SetCheckType();
                else if (_isGivingCheck<MapT>(movePos, fullMap ^ startField, SwapColor(figBoardIndex > wKingIndex)))
                    mv.SetCheckType();

                results.Push(stack, mv);
            }
            if constexpr (promotePawns)
                // upgrading pawn case
            {
                static constexpr size_t startInd = queensIndex;
                static constexpr size_t limitInd = queensIndex - 1;

                // iterating through upgradable pieces
                for (size_t i = startInd; i > limitInd; --i) {
                    const auto targetBoard = _board.MovingColor * BitBoardsPerCol + i;

                    cuda_Move mv{};

                    // preparing basic move info
                    mv.SetStartField(ExtractMsbPos(startField));
                    mv.SetStartBoardIndex(figBoardIndex);
                    mv.SetTargetField(movePos);
                    mv.SetTargetBoardIndex(targetBoard);
                    mv.SetKilledBoardIndex(attackedFigBoardIndex);
                    mv.SetKilledFigureField(movePos);
                    mv.SetElPassantField(InvalidElPassantField);
                    mv.SetCasltingRights(castlings);
                    mv.SetMoveType(CaptureFlag | PromoFlag | PromoFlags[i]);

                    if (_isPromotingPawnGivingCheck<MapT>(movePos, fullMap ^ startField, targetBoard))
                        mv.SetCheckType();

                    results.Push(stack, mv);
                }
            }

            attackingMoves ^= moveBoard;
        }
    }

    static constexpr __uint64_t KING_NO_BLOCKED_MAP = (~static_cast<__uint64_t>(0));

    // TODO: test copying all old castlings
    template<bool GenOnlyTacticalMoves, bool GeneratePseudoMoves = false>
    __device__ void
    _processPlainKingMoves(payload &results, __uint64_t blockedFigMap, __uint64_t allyMap, __uint64_t enemyMap) {
        assert(allyMap != 0 && "Ally map is empty!");
        assert(enemyMap != 0 && "Enemy map is empty!");

        static constexpr size_t CastlingPerColor = 2;

        // generating moves
        const __uint64_t kingMoves = KingMap::GetMoves(_board.GetKingMsbPos(_board.MovingColor)) &
                                     (GeneratePseudoMoves ? KING_NO_BLOCKED_MAP : ~blockedFigMap) &
                                     ~allyMap &
                                     // NOTE: when we do not use blocked fig map we should block kings to prevent attacking themselves
                                     (GeneratePseudoMoves ? ~KingMap::GetMoves(
                                             _board.GetKingMsbPos(SwapColor(_board.MovingColor))) : KING_NO_BLOCKED_MAP);


        __uint64_t attackingMoves = kingMoves & enemyMap;
        [[maybe_unused]] __uint64_t nonAttackingMoves = kingMoves ^ attackingMoves;

        // preparing variables
        auto castlings = _board.Castlings;
        SetBitBoardBit(castlings, _board.MovingColor * CastlingsPerColor + KingCastlingIndex, false);
        SetBitBoardBit(castlings, _board.MovingColor * CastlingsPerColor + QueenCastlingIndex, false);

        const int oldKingPos = ExtractMsbPos(_board.BitBoards[_board.MovingColor * BitBoardsPerCol + kingIndex]);

        // processing simple non-attacking moves
        if constexpr (!GenOnlyTacticalMoves)
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

            // finding an attacked figure
            const size_t attackedFigBoardIndex = GetIndexOfContainingBitBoard(newKingBoard, SwapColor(_board.MovingColor));

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(oldKingPos);
            mv.SetStartBoardIndex(_board.MovingColor * BitBoardsPerCol + kingIndex);
            mv.SetTargetField(newPos);
            mv.SetTargetBoardIndex(_board.MovingColor * BitBoardsPerCol + kingIndex);
            mv.SetKilledBoardIndex(attackedFigBoardIndex);
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

        for (size_t i = 0; i < CastlingsPerColor; ++i)
            if (const size_t castlingIndex = _board.MovingColor * CastlingsPerColor + i;
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
};

#endif // MOVEGENERATOR_CUH
