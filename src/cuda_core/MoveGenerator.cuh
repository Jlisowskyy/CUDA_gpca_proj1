//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef MOVEGENERATOR_CUH
#define MOVEGENERATOR_CUH

#include "Helpers.cuh"

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
        ASSERT(false, "Invalid pawn map type detected!"); \
    }                           \

__device__ static constexpr uint16_t PromoFlags[]{
        0, KnightFlag, BishopFlag, RookFlag, QueenFlag,
};

template<__uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE, __uint32_t STACK_SIZE = UINT32_MAX>
class MoveGenerator : ChessMechanics<NUM_BOARDS> {
    using ChessMechanics<NUM_BOARDS>::GetColBitMap;
    using ChessMechanics<NUM_BOARDS>::GetBlockedFieldBitMap;
    using ChessMechanics<NUM_BOARDS>::GetPinnedFigsMap;
    using ChessMechanics<NUM_BOARDS>::_boardFetcher;
    using ChessMechanics<NUM_BOARDS>::_moveGenData;
    using ChessMechanics<NUM_BOARDS>::GetFullBitMap;
    using ChessMechanics<NUM_BOARDS>::GenerateAllowedTilesForPrecisedPinnedFig;
    using ChessMechanics<NUM_BOARDS>::GetIndexOfContainingBitBoard;
    using ChessMechanics<NUM_BOARDS>::GetBlockedFieldBitMapSplit;
    using ChessMechanics<NUM_BOARDS>::GetAllowedTilesWhenCheckedByNonSliding;

    using _fetcher_t = cuda_PackedBoard<NUM_BOARDS>::BoardFetcher;

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
    Stack<cuda_Move> &stack;

    // ------------------------------
    // Class Creation
    // ------------------------------

    MoveGenerator() = delete;

    FAST_DCALL explicit MoveGenerator(const _fetcher_t &fetcher, Stack<cuda_Move> &s, MoveGenDataMem *md = nullptr)
            : ChessMechanics<NUM_BOARDS>(fetcher, md), stack(s) {}

    MoveGenerator(MoveGenerator &other) = delete;

    MoveGenerator &operator=(MoveGenerator &) = delete;

    ~MoveGenerator() = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    FAST_DCALL void GetMovesFast() {
        // Prepare crucial components and additionally detect whether we are at check and which figure type attacks king
        const __uint64_t fullMap = GetFullBitMap();
        const auto [blockedFigMap, checksCount, wasCheckedBySimple] = GetBlockedFieldBitMap(fullMap);

        ASSERT(blockedFigMap != 0, "Blocked fig map must at least contains fields controlled by king!");
        ASSERT(checksCount <= 2,
               "We consider only 3 states: no-check, single-check, double-check -> invalid result was returned!");

        if (checksCount == 2) {
            _doubleCheckGen(blockedFigMap);
            return;
        }

        _upTo1Check(fullMap, blockedFigMap, checksCount == 1 ? ASSUME_CHECK : EMPTY_FLAGS, wasCheckedBySimple);
    }

    FAST_DCALL void GetMovesSplit(const __uint32_t figIdx) {
        // Prepare crucial components and additionally detect whether we are at check and which figure type attacks king
        const __uint64_t fullMap = GetFullBitMap();
        const auto [blockedFigMap, checksCount, wasCheckedBySimple] = GetBlockedFieldBitMap(fullMap);

        ASSERT(blockedFigMap != 0, "Blocked fig map must at least contains fields controlled by king!");
        ASSERT(checksCount <= 2,
               "We consider only 3 states: no-check, single-check, double-check -> invalid result was returned!");

        if (checksCount == 2) {
            if (figIdx == 0) {
                _doubleCheckGen(blockedFigMap);
            }
            return;
        }

        _upTo1CheckSplit(figIdx, fullMap, blockedFigMap, checksCount == 1 ? ASSUME_CHECK : EMPTY_FLAGS,
                         wasCheckedBySimple);
    }

    [[nodiscard]] __device__ __uint64_t CountMoves(int depth, void *ptr) {
        _fetcher_t fetcher = _boardFetcher;
        return CountMovesRecursive(fetcher, ptr, 0, depth);
    }

    [[nodiscard]] __device__ __uint64_t
    CountMovesRecursive(_fetcher_t &fetcher, void *ptr, int curDepth, int maxDepth) {
        if (curDepth == maxDepth) {
            return 1;
        }

        Stack<cuda_Move> localStack((cuda_Move *) (ptr) + curDepth * DEFAULT_STACK_SIZE);

        MoveGenerator<NUM_BOARDS> mGen{fetcher, localStack};
        mGen.GetMovesFast();

        if ((maxDepth - curDepth) == 1) {
            return localStack.Size();
        }

        __uint64_t sum{};

        VolatileBoardData data(fetcher.Castlings(), fetcher.ElPassantField());
        for (__uint32_t i = 0; i < localStack.Size(); ++i) {
            cuda_Move::MakeMove<NUM_BOARDS>(localStack[i], fetcher);
            sum += CountMovesRecursive(fetcher, ptr, curDepth + 1, maxDepth);
            cuda_Move::UnmakeMove<NUM_BOARDS>(localStack[i], fetcher, data);
        }

        return sum;
    }

    using ChessMechanics<NUM_BOARDS>::IsCheck;

    // ------------------------------
    // private methods
    // ------------------------------

private:

    FAST_DCALL void _upTo1CheckSplit(const __uint32_t figIdx, __uint64_t fullMap, __uint64_t blockedFigMap,
                                     __uint32_t flags, bool wasCheckedBySimpleFig = false) {
        ASSERT(fullMap != 0, "Full map is empty!");
        static constexpr __uint64_t UNUSED = 0;

        const auto [pinnedFigsMap, allowedTilesMap] = [&]() -> thrust::pair<__uint64_t, __uint64_t> {
            if (!IsFlagOn(flags, ASSUME_CHECK)) {
                return GetPinnedFigsMap<ChessMechanics<NUM_BOARDS>::PinnedFigGen::WoutAllowedTiles>(
                        _boardFetcher.MovingColor(), fullMap);
            }

            if (!wasCheckedBySimpleFig) {
                return GetPinnedFigsMap<ChessMechanics<NUM_BOARDS>::PinnedFigGen::WAllowedTiles>(
                        _boardFetcher.MovingColor(), fullMap);
            }

            const auto [rv, _] = GetPinnedFigsMap<ChessMechanics<NUM_BOARDS>::PinnedFigGen::WoutAllowedTiles>(
                    _boardFetcher.MovingColor(), fullMap);

            return {rv, GetAllowedTilesWhenCheckedByNonSliding()};
        }();

        // helping variable preparation
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_boardFetcher.MovingColor()));
        const __uint64_t allyMap = GetColBitMap(_boardFetcher.MovingColor());

        // Specific figure processing
        switch (figIdx) {
            case PAWN_INDEX:
                if (_boardFetcher.MovingColor() == WHITE) {
                    if (!_processPawnMoves<WhitePawnMap>(
                            enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), allowedTilesMap
                    )) {
                        assert(STACK_SIZE != UINT32_MAX);
                        return;
                    }
                } else {
                    if (!_processPawnMoves<BlackPawnMap>(
                            enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), allowedTilesMap
                    )) {
                        assert(STACK_SIZE != UINT32_MAX);
                        return;
                    }
                }
                break;
            case KNIGHT_INDEX:
                if (!_processFigMoves<KnightMap>(
                        enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
                )) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }
                break;
            case BISHOP_INDEX:
                if (!_processFigMoves<BishopMap>(
                        enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
                )) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }
                break;
            case ROOK_INDEX:
                if (!_processFigMoves<RookMap>(
                        enemyMap, allyMap, pinnedFigsMap,
                        (IsFlagOn(flags, ASSUME_CHECK) ? ASSUME_CHECK : CHECK_CASTLINGS),
                        UNUSED, allowedTilesMap
                )) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }
                break;
            case QUEEN_INDEX:
                if (!_processFigMoves<QueenMap>(
                        enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
                )) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }
                break;
            case KING_INDEX:
                if (!_processPlainKingMoves(blockedFigMap, allyMap, enemyMap)) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }

                if (!IsFlagOn(flags, ASSUME_CHECK)) {
                    _processKingCastlings(blockedFigMap, fullMap);
                }
                break;
            default:
                ASSERT(false, "Shit happens");
        }
    }

    FAST_DCALL void _upTo1Check(__uint64_t fullMap, __uint64_t blockedFigMap, __uint32_t flags,
                                bool wasCheckedBySimpleFig = false) {
        ASSERT(fullMap != 0, "Full map is empty!");
        static constexpr __uint64_t UNUSED = 0;

        const auto [pinnedFigsMap, allowedTilesMap] = [&]() -> thrust::pair<__uint64_t, __uint64_t> {
            if (!IsFlagOn(flags, ASSUME_CHECK)) {
                return GetPinnedFigsMap<ChessMechanics<NUM_BOARDS>::PinnedFigGen::WoutAllowedTiles>(
                        _boardFetcher.MovingColor(), fullMap);
            }

            if (!wasCheckedBySimpleFig) {
                return GetPinnedFigsMap<ChessMechanics<NUM_BOARDS>::PinnedFigGen::WAllowedTiles>(
                        _boardFetcher.MovingColor(), fullMap);
            }

            const auto [rv, _] = GetPinnedFigsMap<ChessMechanics<NUM_BOARDS>::PinnedFigGen::WoutAllowedTiles>(
                            _boardFetcher.MovingColor(), fullMap);

            return {rv, GetAllowedTilesWhenCheckedByNonSliding()};
        }();

        // helping variable preparation
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_boardFetcher.MovingColor()));
        const __uint64_t allyMap = GetColBitMap(_boardFetcher.MovingColor());

        // Specific figure processing
        if (!_processFigMoves<KnightMap>(
                enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
        )) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (!_processFigMoves<BishopMap>(
                enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
        )) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (!_processFigMoves<RookMap>(
                enemyMap, allyMap, pinnedFigsMap,
                (IsFlagOn(flags, ASSUME_CHECK) ? ASSUME_CHECK : CHECK_CASTLINGS),
                UNUSED, allowedTilesMap
        )) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (!_processFigMoves<QueenMap>(
                enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), UNUSED, allowedTilesMap
        )) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (_boardFetcher.MovingColor() == WHITE) {
            if (!_processPawnMoves<WhitePawnMap>(
                    enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), allowedTilesMap
            )) {
                assert(STACK_SIZE != UINT32_MAX);
                return;
            }
        } else {
            if (!_processPawnMoves<BlackPawnMap>(
                    enemyMap, allyMap, pinnedFigsMap, ExtractFlag(flags, ASSUME_CHECK), allowedTilesMap
            )) {
                assert(STACK_SIZE != UINT32_MAX);
                return;
            }
        }

        if (!_processPlainKingMoves(blockedFigMap, allyMap, enemyMap)) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (!IsFlagOn(flags, ASSUME_CHECK)) {
            _processKingCastlings(blockedFigMap, fullMap);
        }
    }

    FAST_DCALL void _doubleCheckGen(__uint64_t blockedFigMap) {
        const __uint64_t allyMap = GetColBitMap(_boardFetcher.MovingColor());
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_boardFetcher.MovingColor()));
        _processPlainKingMoves(blockedFigMap, allyMap, enemyMap);
    }

    template<class MapT>
    FAST_DCALL bool _processPawnMoves(
            __uint64_t enemyMap, __uint64_t allyMap, __uint64_t pinnedFigMap, __uint32_t flags,
            __uint64_t allowedMoveFilter = 0
    ) {
        GET_PAWN_FIELD(PromotingMask);

        const __uint64_t promotingPawns = _boardFetcher.BitBoard(MapT::GetBoardIndex(0)) & PromotingMask;
        const __uint64_t nonPromotingPawns = _boardFetcher.BitBoard(MapT::GetBoardIndex(0)) ^ promotingPawns;

        if (!_processFigMoves<MapT>(
                enemyMap, allyMap, pinnedFigMap,
                CONSIDER_EL_PASSANT | SELECT_FIGURES | ExtractFlag(flags, ASSUME_CHECK),
                nonPromotingPawns, allowedMoveFilter
        )) {
            return false;
        }

        // During quiesce search we should also check all promotions so GenOnlyTacticalMoves is false
        if (promotingPawns) {
            if (!_processFigMoves<MapT>(
                    enemyMap, allyMap, pinnedFigMap, SELECT_FIGURES | ExtractFlag(flags, ASSUME_CHECK) | PROMOTE_PAWNS,
                    promotingPawns, allowedMoveFilter
            )) {
                return false;
            }
        }

        if (!_processElPassantMoves<MapT>(
                allyMap | enemyMap, pinnedFigMap, IsFlagOn(flags, ASSUME_CHECK), allowedMoveFilter
        )) {
            return false;
        }

        return true;
    }

    // TODO: Consider different solution?
    template<class MapT>
    FAST_DCALL bool _processElPassantMoves(
            __uint64_t fullMap, __uint64_t pinnedFigMap, bool isCheck, __uint64_t allowedMoveFilter = 0
    ) {
        ASSERT(fullMap != 0, "Full map is empty!");

        if (_boardFetcher.ElPassantField() == INVALID_EL_PASSANT_BIT_BOARD) {
            return true;
        }

        // calculation preparation
        const __uint64_t suspectedFields = MapT::GetElPassantSuspectedFields(_boardFetcher.ElPassantField());
        const __uint32_t enemyCord = SwapColor(_boardFetcher.MovingColor()) * BIT_BOARDS_PER_COLOR;
        const __uint64_t enemyRookFigs =
                _boardFetcher.BitBoard(enemyCord + QUEEN_INDEX) | _boardFetcher.BitBoard(enemyCord + ROOK_INDEX);
        __uint64_t possiblePawnsToMove = _boardFetcher.BitBoard(MapT::GetBoardIndex(0)) & suspectedFields;

        GET_PAWN_FIELD(EnemyElPassantMask);

        while (possiblePawnsToMove) {
            const __uint64_t pawnMap = cuda_MaxMsbPossible >> ExtractMsbPos(possiblePawnsToMove);

            // checking whether move would affect horizontal line attacks on king
            const __uint64_t processedPawns = pawnMap | _boardFetcher.ElPassantField();
            const __uint64_t cleanedFromPawnsMap = fullMap ^ processedPawns;

            if (const __uint64_t kingHorizontalLine =
                        RookMap::GetMoves(_boardFetcher.GetKingMsbPos(_boardFetcher.MovingColor()),
                                          cleanedFromPawnsMap) & EnemyElPassantMask;
                    (kingHorizontalLine & enemyRookFigs) != 0) {
                return true;
            }

            const __uint64_t moveMap = MapT::GetElPassantMoveField(_boardFetcher.ElPassantField());

            // checking whether moving some pawns would undercover king on some line
            if ((processedPawns & pinnedFigMap) != 0) {
                // two separate situations that's need to be considered, every pawn that participate in el passant move
                // should be unblocked on specific lines

                if ((pawnMap & pinnedFigMap) != 0 &&
                    (GenerateAllowedTilesForPrecisedPinnedFig(pawnMap, fullMap) & moveMap) == 0) {
                    possiblePawnsToMove ^= pawnMap;
                    continue;
                }

                if ((GenerateAllowedTilesForPrecisedPinnedFig(_boardFetcher.ElPassantField(), fullMap) & moveMap) ==
                    0) {
                    possiblePawnsToMove ^= pawnMap;
                    continue;
                }
            }

            // When king is checked only if move is going to allow tile el passant is correct
            if (isCheck && (moveMap & allowedMoveFilter) == 0 &&
                (_boardFetcher.ElPassantField() & allowedMoveFilter) == 0) {
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
            mv.SetKilledFigureField(ExtractMsbPos(_boardFetcher.ElPassantField()));
            mv.SetElPassantField(INVALID_EL_PASSANT_FIELD);
            mv.SetCastlingRights(_boardFetcher.Castlings());
            mv.SetMoveType(CaptureFlag);

            if (!stack.Push<STACK_SIZE>(mv)) {
                return false;
            }

            possiblePawnsToMove ^= pawnMap;
        }

        return true;
    }

    // TODO: Compare with simple if searching loop
    // TODO: propagate checkForCastling?
    template<class MapT>
    __device__ bool _processFigMoves(
            __uint64_t enemyMap, __uint64_t allyMap, __uint64_t pinnedFigMap, __uint32_t flags,
            __uint64_t figureSelector = 0, __uint64_t allowedMoveSelector = 0
    ) {
        ASSERT(enemyMap != 0, "Enemy map is empty!");
        ASSERT(allyMap != 0, "Ally map is empty!");

        const __uint64_t fullMap = enemyMap | allyMap;
        __uint64_t pinnedOnes = pinnedFigMap & _boardFetcher.BitBoard(MapT::GetBoardIndex(_boardFetcher.MovingColor()));
        __uint64_t unpinnedOnes = _boardFetcher.BitBoard(MapT::GetBoardIndex(_boardFetcher.MovingColor())) ^ pinnedOnes;

        // applying filter if needed
        if (IsFlagOn(flags, SELECT_FIGURES)) {
            pinnedOnes &= figureSelector;
            unpinnedOnes &= figureSelector;
        }

        // saving results of previous el passant field, used only when the figure is not a pawn

        // processing unpinned moves
        while (unpinnedOnes) {
            // processing moves
            const __uint32_t figPos = ExtractMsbPos(unpinnedOnes);
            const __uint64_t figBoard = cuda_MaxMsbPossible >> figPos;

            // selecting allowed moves if in check
            __uint64_t figMoves = MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap;

            if (IsFlagOn(flags, ASSUME_CHECK)) {
                figMoves &= allowedMoveSelector;
            }

            // Performing checks for castlings
            __uint32_t updatedCastlings = _boardFetcher.Castlings();
            if (IsFlagOn(flags, CHECK_CASTLINGS)) {
                SetBitBoardBit(updatedCastlings, RookMap::GetMatchingCastlingIndex<NUM_BOARDS>(_boardFetcher, figBoard),
                               false);
            }

            // preparing moves
            const __uint64_t attackMoves = figMoves & enemyMap;
            [[maybe_unused]] const __uint64_t nonAttackingMoves = figMoves ^ attackMoves;

            // processing move consequences
            if (!_processNonAttackingMoves(
                    nonAttackingMoves, MapT::GetBoardIndex(_boardFetcher.MovingColor()), figBoard,
                    updatedCastlings, ExtractFlag(flags, CONSIDER_EL_PASSANT) | ExtractFlag(flags, PROMOTE_PAWNS)
            )) {
                return false;
            }

            if (!_processAttackingMoves(
                    attackMoves, MapT::GetBoardIndex(_boardFetcher.MovingColor()), figBoard, updatedCastlings,
                    IsFlagOn(flags, PROMOTE_PAWNS)
            )) {
                return false;
            }

            unpinnedOnes ^= figBoard;
        }

        // if a check is detected, the pinned figure stays in place
        if (IsFlagOn(flags, ASSUME_CHECK))
            return true;

        // processing pinned moves
        // Note: corner Rook possibly applicable to castling cannot be pinned
        while (pinnedOnes) {
            // processing moves
            const __uint32_t figPos = ExtractMsbPos(pinnedOnes);
            const __uint64_t figBoard = cuda_MaxMsbPossible >> figPos;
            const __uint64_t allowedTiles = GenerateAllowedTilesForPrecisedPinnedFig(figBoard, fullMap);
            const __uint64_t figMoves = MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap & allowedTiles;
            // TODO: check applied here?
            // TODO: breaking if there?

            // preparing moves
            const __uint64_t attackMoves = figMoves & enemyMap;
            const __uint64_t nonAttackingMoves = figMoves ^ attackMoves;

            // processing move consequences

            if (!_processNonAttackingMoves(
                    nonAttackingMoves, MapT::GetBoardIndex(_boardFetcher.MovingColor()), figBoard,
                    _boardFetcher.Castlings(),
                    ExtractFlag(flags, CONSIDER_EL_PASSANT) | ExtractFlag(flags, PROMOTE_PAWNS)
            )) {
                return false;
            }

            // TODO: There is exactly one move possible
            if (!_processAttackingMoves(
                    attackMoves, MapT::GetBoardIndex(_boardFetcher.MovingColor()), figBoard,
                    _boardFetcher.Castlings(),
                    IsFlagOn(flags, PROMOTE_PAWNS)
            )) {
                return false;
            }

            pinnedOnes ^= figBoard;
        }

        return true;
    }

    __device__ bool _processNonAttackingMoves(
            __uint64_t nonAttackingMoves, __uint32_t figBoardIndex, __uint64_t startField, __uint32_t castlings,
            __uint32_t flags
    ) {
        ASSERT(figBoardIndex < BIT_BOARDS_COUNT, "Invalid figure cuda_Board index!");

        while (nonAttackingMoves) {
            // extracting moves
            const __uint32_t movePos = ExtractMsbPos(nonAttackingMoves);
            const __uint64_t moveBoard = cuda_MaxMsbPossible >> movePos;

            cuda_Move mv{};

            mv.SetStartField(ExtractMsbPos(startField));
            mv.SetStartBoardIndex(figBoardIndex);
            mv.SetTargetField(movePos);
            mv.SetCastlingRights(castlings);
            mv.SetKilledBoardIndex(SENTINEL_BOARD_INDEX);

            if (IsFlagOn(flags, CONSIDER_EL_PASSANT)) {
                const auto result = (_boardFetcher.MovingColor() == WHITE ?
                                     WhitePawnMap::GetElPassantField(moveBoard, startField) :
                                     BlackPawnMap::GetElPassantField(moveBoard, startField)
                );

                mv.SetElPassantField(result == 0 ? INVALID_EL_PASSANT_FIELD : ExtractMsbPos(result));
            } else {
                mv.SetElPassantField(INVALID_EL_PASSANT_FIELD);
            }

            mv.SetTargetBoardIndex(
                    IsFlagOn(flags, PROMOTE_PAWNS) ? _boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + QUEEN_INDEX
                                                   : figBoardIndex);
            mv.SetMoveType(IsFlagOn(flags, PROMOTE_PAWNS) ? PromoFlag | PromoFlags[QUEEN_INDEX] : 0);

            if (!stack.Push<STACK_SIZE>(mv)) {
                return false;
            }

            nonAttackingMoves ^= moveBoard;
        }

        return true;
    }

    FAST_DCALL bool _processAttackingMoves(
            __uint64_t attackingMoves, __uint32_t figBoardIndex, __uint64_t startField,
            __uint32_t castlings, bool promotePawns
    ) {
        ASSERT(figBoardIndex < BIT_BOARDS_COUNT, "Invalid figure cuda_Board index!");

        while (attackingMoves) {
            // extracting moves
            const __uint32_t movePos = ExtractMsbPos(attackingMoves);
            const __uint64_t moveBoard = cuda_MaxMsbPossible >> movePos;
            const __uint32_t attackedFigBoardIndex = GetIndexOfContainingBitBoard(moveBoard,
                                                                                  SwapColor(
                                                                                          _boardFetcher.MovingColor()));

            cuda_Move mv{};

            mv.SetStartField(ExtractMsbPos(startField));
            mv.SetStartBoardIndex(figBoardIndex);
            mv.SetTargetField(movePos);
            mv.SetKilledBoardIndex(attackedFigBoardIndex);
            mv.SetKilledFigureField(movePos);
            mv.SetElPassantField(INVALID_EL_PASSANT_FIELD);
            mv.SetCastlingRights(castlings);
            mv.SetMoveType(CaptureFlag);
            mv.SetTargetBoardIndex(
                    promotePawns ? _boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + QUEEN_INDEX : figBoardIndex);
            mv.SetMoveType(promotePawns ? PromoFlag | PromoFlags[QUEEN_INDEX] : 0);

            if (!stack.Push<STACK_SIZE>(mv)) {
                return false;
            }

            attackingMoves ^= moveBoard;
        }

        return true;
    }

    __device__ bool
    _processPlainKingMoves(__uint64_t blockedFigMap, __uint64_t allyMap, __uint64_t enemyMap) {
        ASSERT(allyMap != 0, "Ally map is empty!");
        ASSERT(enemyMap != 0, "Enemy map is empty!");

        // generating moves
        const __uint64_t kingMoves = KingMap::GetMoves(_boardFetcher.GetKingMsbPos(_boardFetcher.MovingColor())) &
                                     ~blockedFigMap & ~allyMap;

        __uint64_t attackingMoves = kingMoves & enemyMap;
        __uint64_t nonAttackingMoves = kingMoves ^ attackingMoves;

        // preparing variables
        auto castlings = _boardFetcher.Castlings();
        SetBitBoardBit(castlings, _boardFetcher.MovingColor() * CASTLINGS_PER_COLOR + KING_CASTLING_INDEX, false);
        SetBitBoardBit(castlings, _boardFetcher.MovingColor() * CASTLINGS_PER_COLOR + QUEEN_CASTLING_INDEX, false);

        const __uint32_t oldKingPos = ExtractMsbPos(_boardFetcher.GetFigBoard(_boardFetcher.MovingColor(), KING_INDEX));

        // processing simple non-attacking moves
        while (nonAttackingMoves) {
            // extracting new king position data
            const __uint32_t newPos = ExtractMsbPos(nonAttackingMoves);

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(oldKingPos);
            mv.SetStartBoardIndex(_boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + KING_INDEX);
            mv.SetTargetField(newPos);
            mv.SetTargetBoardIndex(_boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + KING_INDEX);
            mv.SetKilledBoardIndex(SENTINEL_BOARD_INDEX);
            mv.SetElPassantField(INVALID_EL_PASSANT_FIELD);
            mv.SetCastlingRights(castlings);

            if (!stack.Push<STACK_SIZE>(mv)) {
                return false;
            }

            nonAttackingMoves ^= (cuda_MaxMsbPossible >> newPos);
        }

        // processing slightly more complicated attacking moves
        while (attackingMoves) {
            // extracting new king position data
            const __uint32_t newPos = ExtractMsbPos(attackingMoves);
            const __uint64_t newKingBoard = cuda_MaxMsbPossible >> newPos;

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(oldKingPos);
            mv.SetStartBoardIndex(_boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + KING_INDEX);
            mv.SetTargetField(newPos);
            mv.SetTargetBoardIndex(_boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + KING_INDEX);
            mv.SetKilledBoardIndex(GetIndexOfContainingBitBoard(newKingBoard, SwapColor(_boardFetcher.MovingColor())));
            mv.SetKilledFigureField(newPos);
            mv.SetElPassantField(INVALID_EL_PASSANT_FIELD);
            mv.SetCastlingRights(castlings);
            mv.SetMoveType(CaptureFlag);

            if (!stack.Push<STACK_SIZE>(mv)) {
                return false;
            }

            attackingMoves ^= newKingBoard;
        }

        return true;
    }

    // TODO: simplify ifs??
    // TODO: cleanup left castling available when rook is dead then propagate no castling checking?
    FAST_DCALL bool _processKingCastlings(__uint64_t blockedFigMap, __uint64_t fullMap) {
        ASSERT(fullMap != 0, "Full map is empty!");

        for (__uint32_t i = 0; i < CASTLINGS_PER_COLOR; ++i) {
            if (const __uint32_t castlingIndex = _boardFetcher.MovingColor() * CASTLINGS_PER_COLOR + i;
                    _boardFetcher.GetCastlingRight(castlingIndex) &&
                    (CASTLING_ROOK_MAPS[castlingIndex] &
                            _boardFetcher.BitBoard(_boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + ROOK_INDEX)) !=
                    0 &&
                    (CASTLING_SENSITIVE_FIELDS[castlingIndex] & blockedFigMap) == 0 &&
                    (CASTLING_TOUCHED_FIELDS[castlingIndex] & fullMap) == 0) {

                auto castlings = _boardFetcher.Castlings();
                SetBitBoardBit(castlings, _boardFetcher.MovingColor() * CASTLINGS_PER_COLOR + KING_CASTLING_INDEX,
                               false);
                SetBitBoardBit(castlings, _boardFetcher.MovingColor() * CASTLINGS_PER_COLOR + QUEEN_CASTLING_INDEX,
                               false);

                cuda_Move mv{};

                // preparing basic move info
                mv.SetStartField(ExtractMsbPos(DEFAULT_KING_BOARDS[_boardFetcher.MovingColor()]));
                mv.SetStartBoardIndex(_boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + KING_INDEX);
                mv.SetTargetField(CASTLING_NEW_KING_POS[castlingIndex]);
                mv.SetTargetBoardIndex(_boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + KING_INDEX);
                mv.SetKilledBoardIndex(_boardFetcher.MovingColor() * BIT_BOARDS_PER_COLOR + ROOK_INDEX);
                mv.SetKilledFigureField(ExtractMsbPos(CASTLING_ROOK_MAPS[castlingIndex]));
                mv.SetElPassantField(INVALID_EL_PASSANT_FIELD);
                mv.SetCastlingRights(castlings);
                mv.SetCastlingType(1 + castlingIndex); // God only knows why I made a sentinel at index 0 at this moment
                mv.SetMoveType(CastlingFlag);

                if (!stack.Push<STACK_SIZE>(mv)) {
                    return false;
                }
            }
        }

        return true;
    }
};

#endif // MOVEGENERATOR_CUH
