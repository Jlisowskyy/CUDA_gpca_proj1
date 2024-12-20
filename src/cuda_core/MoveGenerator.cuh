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
    uint64_t param{}; \
    \
    if constexpr (std::is_same<MapT, WhitePawnMap>::value) { \
        param = WhitePawnMapConstants::param; \
    } else if constexpr (std::is_same<MapT, BlackPawnMap>::value) { \
        param = BlackPawnMapConstants::param; \
    } else { \
        ASSERT(false, "Invalid pawn map type detected!"); \
    }
__device__ static constexpr uint16_t PromoFlags[]{
    0, KnightFlag, BishopFlag, RookFlag, QueenFlag,
};

template<uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE, uint32_t STACK_SIZE = UINT32_MAX>
class MoveGenerator : ChessMechanics<NUM_BOARDS> {
    using ChessMechanics<NUM_BOARDS>::GetColBitMap;
    using ChessMechanics<NUM_BOARDS>::GetBlockedFieldBitMap;
    using ChessMechanics<NUM_BOARDS>::GetPinnedFigsMap;
    using ChessMechanics<NUM_BOARDS>::_boardFetcher;
    using ChessMechanics<NUM_BOARDS>::_moveGenData;
    using ChessMechanics<NUM_BOARDS>::GetFullBitMap;
    using ChessMechanics<NUM_BOARDS>::GenerateAllowedTilesForPrecisedPinnedFig;
    using ChessMechanics<NUM_BOARDS>::GetIndexOfContainingBitBoard;
    using ChessMechanics<NUM_BOARDS>::GetAllowedTilesWhenCheckedByNonSliding;

    using _fetcher_t = typename cuda_PackedBoard<NUM_BOARDS>::BoardFetcher;

    enum MoveGenFlags : uint32_t {
        EMPTY_FLAGS = 0,
        CHECK_CASTLINGS = 1,
        PROMOTE_PAWNS = 2,
        SELECT_FIGURES = 4,
        ASSUME_CHECK = 8,
        CONSIDER_EL_PASSANT = 16,
    };

    FAST_DCALL_ALWAYS static constexpr uint32_t ExtractFlag(uint32_t flags, MoveGenFlags flag) {
        return flags & flag;
    }

    FAST_DCALL_ALWAYS static constexpr bool IsFlagOn(uint32_t flags, MoveGenFlags flag) {
        return ExtractFlag(flags, flag) != 0;
    }

public:
    Stack<cuda_Move> &stack;

    // ------------------------------
    // Class Creation
    // ------------------------------

    MoveGenerator() = delete;

    FAST_DCALL explicit MoveGenerator(const _fetcher_t &fetcher, Stack<cuda_Move> &s, MoveGenDataMem *md = nullptr)
        : ChessMechanics<NUM_BOARDS>(fetcher, md), stack(s) {
    }

    MoveGenerator(MoveGenerator &other) = delete;

    MoveGenerator &operator=(MoveGenerator &) = delete;

    ~MoveGenerator() = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    using ChessMechanics<NUM_BOARDS>::EvalBoardsNoMoves;


    FAST_DCALL void GetMovesFast() {
        // Prepare crucial components and additionally detect whether we are at check and which figure type attacks king
        const uint32_t movingColor = _boardFetcher.MovingColor();
        const uint64_t enemyMap = GetColBitMap(SwapColor(movingColor));
        const uint64_t allyMap = GetColBitMap(movingColor);
        const auto [blockedFigMap, checksCount, wasCheckedBySimple] = GetBlockedFieldBitMap(movingColor,
            enemyMap | allyMap);

        ASSERT(blockedFigMap != 0, "Blocked fig map must at least contains fields controlled by king!");
        ASSERT(checksCount <= 2,
               "We consider only 3 states: no-check, single-check, double-check -> invalid result was returned!");

        if (checksCount == 2) {
            _doubleCheckGen(movingColor, blockedFigMap);
            return;
        }

        _upTo1Check(movingColor, allyMap, enemyMap, blockedFigMap, checksCount == 1 ? ASSUME_CHECK : EMPTY_FLAGS,
                    wasCheckedBySimple);
    }

    FAST_DCALL void GetMovesSplit(const uint32_t figIdx) {
        // Prepare crucial components and additionally detect whether we are at check and which figure type attacks king
        const uint32_t movingColor = _boardFetcher.MovingColor();
        const uint64_t enemyMap = GetColBitMap(SwapColor(movingColor));
        const uint64_t allyMap = GetColBitMap(movingColor);
        const auto [blockedFigMap, checksCount, wasCheckedBySimple] = GetBlockedFieldBitMap(movingColor,
            allyMap | enemyMap);

        ASSERT(blockedFigMap != 0, "Blocked fig map must at least contains fields controlled by king!");
        ASSERT(checksCount <= 2,
               "We consider only 3 states: no-check, single-check, double-check -> invalid result was returned!");

        if (checksCount == 2) {
            if (figIdx == 0) {
                _doubleCheckGen(movingColor, blockedFigMap);
            }
            return;
        }

        _upTo1CheckSplit(movingColor, figIdx, allyMap, enemyMap, blockedFigMap,
                         checksCount == 1 ? ASSUME_CHECK : EMPTY_FLAGS,
                         wasCheckedBySimple);
    }

    [[nodiscard]] __device__ uint64_t CountMoves(int depth, void *ptr) {
        _fetcher_t fetcher = _boardFetcher;
        return CountMovesRecursive(fetcher, ptr, 0, depth);
    }

    [[nodiscard]] __device__ uint64_t CountMovesSplit(uint32_t figIdx, int depth, void *ptr) {
        _fetcher_t fetcher = _boardFetcher;
        return CountMovesRecursiveSplit(figIdx, fetcher, ptr, 0, depth);
    }

    [[nodiscard]] __device__ uint64_t
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

        uint64_t sum{};

        VolatileBoardData data(fetcher.ElPassantField(), fetcher.HalfMoves(), fetcher.Castlings());
        for (uint32_t i = 0; i < localStack.Size(); ++i) {
            cuda_Move::MakeMove<NUM_BOARDS>(localStack[i], fetcher);
            sum += CountMovesRecursive(fetcher, ptr, curDepth + 1, maxDepth);
            cuda_Move::UnmakeMove<NUM_BOARDS>(localStack[i], fetcher, data);
        }

        return sum;
    }

    [[nodiscard]] __device__ uint64_t
    CountMovesRecursiveSplit(uint32_t figIdx, _fetcher_t &fetcher, void *ptr, int curDepth, int maxDepth) {
        if (curDepth == maxDepth) {
            return 1;
        }

        Stack<cuda_Move> localStack((void *) (ptr) + curDepth * DEFAULT_STACK_SIZE, false);

        if (figIdx == 0) {
            localStack.Clear();
        }

        __syncthreads();

        MoveGenerator<NUM_BOARDS> mGen{fetcher, localStack};
        mGen.GetMovesFast();

        __syncthreads();

        if ((maxDepth - curDepth) == 1) {
            return localStack.Size();
        }

        uint64_t sum{};

        VolatileBoardData data(fetcher.ElPassantField(), fetcher.HalfMoves(), fetcher.Castlings());
        for (uint32_t i = 0; i < localStack.Size(); ++i) {
            if (figIdx == 0) {
                cuda_Move::MakeMove<NUM_BOARDS>(localStack[i], fetcher);
            }

            __syncthreads();
            sum += CountMovesRecursive(fetcher, ptr, curDepth + 1, maxDepth);

            if (figIdx == 0) {
                cuda_Move::UnmakeMove<NUM_BOARDS>(localStack[i], fetcher, data);
            }

            __syncthreads();
        }

        return sum;
    }

    // ------------------------------
    // private methods
    // ------------------------------

private:
    FAST_DCALL void _upTo1CheckSplit(const uint32_t movingColor, const uint32_t figIdx, const uint64_t allyMap,
                                     const uint64_t enemyMap, const uint64_t blockedFigMap,
                                     const uint32_t flags, bool wasCheckedBySimpleFig = false) {
        ASSERT(allyMap != 0 && enemyMap != 0, "Full map is empty!");
        static constexpr uint64_t UNUSED = 0;

        const auto [pinnedFigsMap, allowedTilesMap] = [&]() -> thrust::pair<uint64_t, uint64_t> {
            if (!IsFlagOn(flags, ASSUME_CHECK)) {
                return GetPinnedFigsMap < ChessMechanics<NUM_BOARDS>::PinnedFigGen::WoutAllowedTiles > (
                           movingColor, allyMap | enemyMap);
            }

            if (!wasCheckedBySimpleFig) {
                return GetPinnedFigsMap < ChessMechanics<NUM_BOARDS>::PinnedFigGen::WAllowedTiles > (
                           movingColor, allyMap | enemyMap);
            }

            const auto [rv, _] = GetPinnedFigsMap < ChessMechanics<NUM_BOARDS>::PinnedFigGen::WoutAllowedTiles > (
                                     movingColor, allyMap | enemyMap);

            return {rv, GetAllowedTilesWhenCheckedByNonSliding(movingColor)};
        }();

        // Specific figure processing
        switch (figIdx) {
            case PAWN_INDEX:
                if (movingColor == WHITE) {
                    if (!_processPawnMoves<WhitePawnMap>(
                        movingColor, enemyMap, allyMap, pinnedFigsMap, flags, allowedTilesMap
                    )) {
                        assert(STACK_SIZE != UINT32_MAX);
                        return;
                    }
                } else {
                    if (!_processPawnMoves<BlackPawnMap>(
                        movingColor, enemyMap, allyMap, pinnedFigsMap, flags, allowedTilesMap
                    )) {
                        assert(STACK_SIZE != UINT32_MAX);
                        return;
                    }
                }
                break;
            case KNIGHT_INDEX:
                if (!_processFigMoves<KnightMap>(
                    movingColor, enemyMap, allyMap, pinnedFigsMap, flags, UNUSED, allowedTilesMap
                )) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }
                break;
            case BISHOP_INDEX:
                if (!_processFigMoves<BishopMap>(
                    movingColor, enemyMap, allyMap, pinnedFigsMap, flags, UNUSED, allowedTilesMap
                )) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }
                break;
            case ROOK_INDEX:
                if (!_processFigMoves<RookMap>(
                    movingColor, enemyMap, allyMap, pinnedFigsMap, flags | CHECK_CASTLINGS, UNUSED, allowedTilesMap
                )) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }
                break;
            case QUEEN_INDEX:
                if (!_processFigMoves<QueenMap>(
                    movingColor, enemyMap, allyMap, pinnedFigsMap, flags, UNUSED, allowedTilesMap
                )) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }
                break;
            case KING_INDEX:
                if (!_processPlainKingMoves(movingColor, blockedFigMap, allyMap, enemyMap)) {
                    assert(STACK_SIZE != UINT32_MAX);
                    return;
                }

                if (!IsFlagOn(flags, ASSUME_CHECK)) {
                    _processKingCastlings(movingColor, blockedFigMap, allyMap | enemyMap);
                }
                break;
            default:
                ASSERT(false, "Shit happens");
        }
    }

    FAST_DCALL void
    _upTo1Check(const uint32_t movingColor, const uint64_t allyMap, const uint64_t enemyMap,
                const uint64_t blockedFigMap,
                const uint32_t flags, const bool wasCheckedBySimpleFig = false) {
        ASSERT(allyMap != 0 && enemyMap != 0, "Full map is empty!");
        static constexpr uint64_t UNUSED = 0;

        const auto [pinnedFigsMap, allowedTilesMap] = [&]() -> thrust::pair<uint64_t, uint64_t> {
            if (!IsFlagOn(flags, ASSUME_CHECK)) {
                return GetPinnedFigsMap < ChessMechanics<NUM_BOARDS>::PinnedFigGen::WoutAllowedTiles > (
                           movingColor, allyMap | enemyMap);
            }

            if (!wasCheckedBySimpleFig) {
                return GetPinnedFigsMap < ChessMechanics<NUM_BOARDS>::PinnedFigGen::WAllowedTiles > (
                           movingColor, allyMap | enemyMap);
            }

            const auto [rv, _] = GetPinnedFigsMap < ChessMechanics<NUM_BOARDS>::PinnedFigGen::WoutAllowedTiles > (
                                     movingColor, allyMap | enemyMap);

            return {rv, GetAllowedTilesWhenCheckedByNonSliding(movingColor)};
        }();

        // Specific figure processing
        if (!_processFigMoves<KnightMap>(
            movingColor, enemyMap, allyMap, pinnedFigsMap, flags, UNUSED, allowedTilesMap
        )) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (!_processFigMoves<BishopMap>(
            movingColor, enemyMap, allyMap, pinnedFigsMap, flags, UNUSED, allowedTilesMap
        )) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (!_processFigMoves<RookMap>(
            movingColor, enemyMap, allyMap, pinnedFigsMap, flags | CHECK_CASTLINGS, UNUSED, allowedTilesMap
        )) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (!_processFigMoves<QueenMap>(
            movingColor, enemyMap, allyMap, pinnedFigsMap, flags, UNUSED, allowedTilesMap
        )) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (movingColor == WHITE) {
            if (!_processPawnMoves<WhitePawnMap>(
                movingColor, enemyMap, allyMap, pinnedFigsMap, flags, allowedTilesMap
            )) {
                assert(STACK_SIZE != UINT32_MAX);
                return;
            }
        } else {
            if (!_processPawnMoves<BlackPawnMap>(
                movingColor, enemyMap, allyMap, pinnedFigsMap, flags, allowedTilesMap
            )) {
                assert(STACK_SIZE != UINT32_MAX);
                return;
            }
        }

        if (!_processPlainKingMoves(movingColor, blockedFigMap, allyMap, enemyMap)) {
            assert(STACK_SIZE != UINT32_MAX);
            return;
        }

        if (!IsFlagOn(flags, ASSUME_CHECK)) {
            _processKingCastlings(movingColor, blockedFigMap, allyMap | enemyMap);
        }
    }

    FAST_DCALL void _doubleCheckGen(const uint32_t movingColor, const uint64_t blockedFigMap) {
        const uint64_t allyMap = GetColBitMap(movingColor);
        const uint64_t enemyMap = GetColBitMap(SwapColor(movingColor));
        _processPlainKingMoves(movingColor, blockedFigMap, allyMap, enemyMap);
    }

    template<class MapT>
    FAST_DCALL bool _processPawnMoves(
        const uint32_t movingColor, const uint64_t enemyMap, const uint64_t allyMap,
        const uint64_t pinnedFigMap, const uint32_t flags,
        const uint64_t allowedMoveFilter = 0
    ) {
        GET_PAWN_FIELD(PromotingMask);

        const uint64_t promotingPawns = _boardFetcher.BitBoard(MapT::GetBoardIndex(0)) & PromotingMask;
        const uint64_t nonPromotingPawns = _boardFetcher.BitBoard(MapT::GetBoardIndex(0)) ^ promotingPawns;

        if (!_processFigMoves<MapT>(
            movingColor, enemyMap, allyMap, pinnedFigMap, CONSIDER_EL_PASSANT | SELECT_FIGURES | flags,
            nonPromotingPawns,
            allowedMoveFilter
        )) {
            return false;
        }

        // During quiesce search we should also check all promotions so GenOnlyTacticalMoves is false
        if (promotingPawns) {
            if (!_processFigMoves<MapT>(
                movingColor, enemyMap, allyMap, pinnedFigMap, SELECT_FIGURES | flags | PROMOTE_PAWNS,
                promotingPawns, allowedMoveFilter
            )) {
                return false;
            }
        }

        if (!_processElPassantMoves<MapT>(
            movingColor, allyMap | enemyMap, pinnedFigMap, IsFlagOn(flags, ASSUME_CHECK), allowedMoveFilter
        )) {
            return false;
        }

        return true;
    }

    // TODO: Possibly should be optimized, but it's not a critical path
    template<class MapT>
    FAST_DCALL bool _processElPassantMoves(
        const uint32_t movingColor, const uint64_t fullMap, const uint64_t pinnedFigMap, const bool isCheck,
        const uint64_t allowedMoveFilter = 0
    ) {
        ASSERT(fullMap != 0, "Full map is empty!");

        if (_boardFetcher.ElPassantField() == INVALID_EL_PASSANT_BIT_BOARD) {
            return true;
        }

        // calculation preparation
        const uint64_t suspectedFields = MapT::GetElPassantSuspectedFields(_boardFetcher.ElPassantField());
        const uint32_t enemyCord = SwapColor(movingColor) * BIT_BOARDS_PER_COLOR;
        const uint64_t enemyRookFigs =
                _boardFetcher.BitBoard(enemyCord + QUEEN_INDEX) | _boardFetcher.BitBoard(enemyCord + ROOK_INDEX);
        uint64_t possiblePawnsToMove = _boardFetcher.BitBoard(MapT::GetBoardIndex(0)) & suspectedFields;

        GET_PAWN_FIELD(EnemyElPassantMask);

        while (possiblePawnsToMove) {
            const uint64_t pawnMap = cuda_MaxMsbPossible >> ExtractMsbPos(possiblePawnsToMove);

            // checking whether move would affect horizontal line attacks on king
            const uint64_t processedPawns = pawnMap | _boardFetcher.ElPassantField();
            const uint64_t cleanedFromPawnsMap = fullMap ^ processedPawns;

            if (const uint64_t kingHorizontalLine =
                        RookMap::GetMoves(_boardFetcher.GetKingMsbPos(movingColor),
                                          cleanedFromPawnsMap) & EnemyElPassantMask;
                (kingHorizontalLine & enemyRookFigs) != 0) {
                return true;
            }

            const uint64_t moveMap = MapT::GetElPassantMoveField(_boardFetcher.ElPassantField());

            // checking whether moving some pawns would undercover king on some line
            if ((processedPawns & pinnedFigMap) != 0) {
                // two separate situations that's need to be considered, every pawn that participate in el passant move
                // should be unblocked on specific lines

                if ((pawnMap & pinnedFigMap) != 0 &&
                    (GenerateAllowedTilesForPrecisedPinnedFig(movingColor, pawnMap, fullMap) & moveMap) == 0) {
                    possiblePawnsToMove ^= pawnMap;
                    continue;
                }

                if ((GenerateAllowedTilesForPrecisedPinnedFig(movingColor, _boardFetcher.ElPassantField(), fullMap) &
                     moveMap) ==
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
    template<class MapT>
    __device__ bool _processFigMoves(
        const uint32_t movingColor, const uint64_t enemyMap, const uint64_t allyMap,
        const uint64_t pinnedFigMap, const uint32_t flags,
        const uint64_t figureSelector = 0, const uint64_t allowedMoveSelector = 0
    ) {
        ASSERT(enemyMap != 0, "Enemy map is empty!");
        ASSERT(allyMap != 0, "Ally map is empty!");

        const uint64_t fullMap = enemyMap | allyMap;
        uint64_t pinnedOnes = pinnedFigMap & _boardFetcher.BitBoard(MapT::GetBoardIndex(movingColor));
        uint64_t unpinnedOnes = _boardFetcher.BitBoard(MapT::GetBoardIndex(movingColor)) ^ pinnedOnes;

        // applying filter if needed
        if (IsFlagOn(flags, SELECT_FIGURES)) {
            pinnedOnes &= figureSelector;
            unpinnedOnes &= figureSelector;
        }

        // processing unpinned moves
        while (unpinnedOnes) {
            // processing moves
            const uint32_t figPos = ExtractMsbPos(unpinnedOnes);
            const uint64_t figBoard = cuda_MaxMsbPossible >> figPos;

            // selecting allowed moves if in check
            uint64_t figMoves = MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap;

            if (IsFlagOn(flags, ASSUME_CHECK)) {
                figMoves &= allowedMoveSelector;
            }

            // Performing checks for castlings
            uint32_t updatedCastlings = _boardFetcher.Castlings();
            if (IsFlagOn(flags, CHECK_CASTLINGS) && !IsFlagOn(flags, ASSUME_CHECK)) {
                SetBitBoardBit(updatedCastlings, RookMap::GetMatchingCastlingIndex<NUM_BOARDS>(_boardFetcher, figBoard),
                               false);
            }

            // preparing moves
            const uint64_t attackMoves = figMoves & enemyMap;
            const uint64_t nonAttackingMoves = figMoves ^ attackMoves;

            // processing move consequences
            if (!_processNonAttackingMoves(
                movingColor, nonAttackingMoves, MapT::GetBoardIndex(movingColor), figBoard,
                updatedCastlings, flags
            )) {
                return false;
            }

            if (!_processAttackingMoves(
                movingColor, attackMoves, MapT::GetBoardIndex(movingColor), figBoard, updatedCastlings,
                IsFlagOn(flags, PROMOTE_PAWNS)
            )) {
                return false;
            }

            unpinnedOnes ^= figBoard;
        }

        // if a check is detected, the pinned figure stays in place
        if (IsFlagOn(flags, ASSUME_CHECK)) {
            return true;
        }

        // processing pinned moves
        // Note: corner Rook possibly applicable to castling cannot be pinned
        while (pinnedOnes) {
            // processing moves
            const uint32_t figPos = ExtractMsbPos(pinnedOnes);
            const uint64_t figBoard = cuda_MaxMsbPossible >> figPos;
            const uint64_t allowedTiles = GenerateAllowedTilesForPrecisedPinnedFig(movingColor, figBoard, fullMap);
            const uint64_t figMoves = MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap & allowedTiles;

            // preparing moves
            const uint64_t attackMoves = figMoves & enemyMap;
            const uint64_t nonAttackingMoves = figMoves ^ attackMoves;

            // processing move consequences
            if (!_processNonAttackingMoves(
                movingColor, nonAttackingMoves, MapT::GetBoardIndex(movingColor), figBoard,
                _boardFetcher.Castlings(),
                flags
            )) {
                return false;
            }

            // TODO: There is exactly one move possible
            if (!_processAttackingMoves(
                movingColor, attackMoves, MapT::GetBoardIndex(movingColor), figBoard,
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
        const uint32_t movingColor, uint64_t nonAttackingMoves, const uint32_t figBoardIndex, const uint64_t startField,
        const uint32_t castlings, const uint32_t flags
    ) {
        ASSERT(figBoardIndex < BIT_BOARDS_COUNT, "Invalid figure cuda_Board index!");

        while (nonAttackingMoves) {
            // extracting moves
            const uint32_t movePos = ExtractMsbPos(nonAttackingMoves);
            const uint64_t moveBoard = cuda_MaxMsbPossible >> movePos;

            cuda_Move mv{};

            mv.SetStartField(ExtractMsbPos(startField));
            mv.SetStartBoardIndex(figBoardIndex);
            mv.SetTargetField(movePos);
            mv.SetCastlingRights(castlings);
            mv.SetKilledBoardIndex(SENTINEL_BOARD_INDEX);

            if (IsFlagOn(flags, CONSIDER_EL_PASSANT)) {
                const auto result = (movingColor == WHITE
                                         ? WhitePawnMap::GetElPassantField(moveBoard, startField)
                                         : BlackPawnMap::GetElPassantField(moveBoard, startField)
                );

                mv.SetElPassantField(result == 0 ? INVALID_EL_PASSANT_FIELD : ExtractMsbPos(result));
            } else {
                mv.SetElPassantField(INVALID_EL_PASSANT_FIELD);
            }

            mv.SetTargetBoardIndex(
                IsFlagOn(flags, PROMOTE_PAWNS)
                    ? movingColor * BIT_BOARDS_PER_COLOR + QUEEN_INDEX
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
        const uint32_t movingColor, uint64_t attackingMoves, const uint32_t figBoardIndex, const uint64_t startField,
        const uint32_t castlings, const bool promotePawns
    ) {
        ASSERT(figBoardIndex < BIT_BOARDS_COUNT, "Invalid figure cuda_Board index!");

        while (attackingMoves) {
            // extracting moves
            const uint32_t movePos = ExtractMsbPos(attackingMoves);
            const uint64_t moveBoard = cuda_MaxMsbPossible >> movePos;
            const uint32_t attackedFigBoardIndex =
                    GetIndexOfContainingBitBoard(moveBoard, SwapColor(movingColor));

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
                promotePawns ? movingColor * BIT_BOARDS_PER_COLOR + QUEEN_INDEX : figBoardIndex);
            mv.SetMoveType(promotePawns ? PromoFlag | PromoFlags[QUEEN_INDEX] : 0);

            if (!stack.Push<STACK_SIZE>(mv)) {
                return false;
            }

            attackingMoves ^= moveBoard;
        }

        return true;
    }

    __device__ bool
    _processPlainKingMoves(const uint32_t movingColor, const uint64_t blockedFigMap, const uint64_t allyMap,
                           const uint64_t enemyMap) {
        ASSERT(allyMap != 0, "Ally map is empty!");
        ASSERT(enemyMap != 0, "Enemy map is empty!");

        // generating moves
        const uint64_t kingMoves = KingMap::GetMoves(_boardFetcher.GetKingMsbPos(movingColor)) &
                                   ~blockedFigMap & ~allyMap;

        uint64_t attackingMoves = kingMoves & enemyMap;
        uint64_t nonAttackingMoves = kingMoves ^ attackingMoves;

        // preparing variables
        auto castlings = _boardFetcher.Castlings();
        SetBitBoardBit(castlings, movingColor * CASTLINGS_PER_COLOR + KING_CASTLING_INDEX, false);
        SetBitBoardBit(castlings, movingColor * CASTLINGS_PER_COLOR + QUEEN_CASTLING_INDEX, false);

        const uint32_t oldKingPos = ExtractMsbPos(_boardFetcher.GetFigBoard(movingColor, KING_INDEX));

        // processing simple non-attacking moves
        while (nonAttackingMoves) {
            // extracting new king position data
            const uint32_t newPos = ExtractMsbPos(nonAttackingMoves);

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(oldKingPos);
            mv.SetStartBoardIndex(movingColor * BIT_BOARDS_PER_COLOR + KING_INDEX);
            mv.SetTargetField(newPos);
            mv.SetTargetBoardIndex(movingColor * BIT_BOARDS_PER_COLOR + KING_INDEX);
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
            const uint32_t newPos = ExtractMsbPos(attackingMoves);
            const uint64_t newKingBoard = cuda_MaxMsbPossible >> newPos;

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(oldKingPos);
            mv.SetStartBoardIndex(movingColor * BIT_BOARDS_PER_COLOR + KING_INDEX);
            mv.SetTargetField(newPos);
            mv.SetTargetBoardIndex(movingColor * BIT_BOARDS_PER_COLOR + KING_INDEX);
            mv.SetKilledBoardIndex(GetIndexOfContainingBitBoard(newKingBoard, SwapColor(movingColor)));
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

    // TODO: Possibly should be optimized, it's not a critical path, but still more relevant than en passant
    FAST_DCALL bool _processKingCastlings(const uint32_t movingColor, const uint64_t blockedFigMap,
                                          const uint64_t fullMap) {
        ASSERT(fullMap != 0, "Full map is empty!");

#pragma unroll
        for (uint32_t i = 0; i < CASTLINGS_PER_COLOR; ++i) {
            if (const uint32_t castlingIndex = movingColor * CASTLINGS_PER_COLOR + i;
                _boardFetcher.GetCastlingRight(castlingIndex) &&
                (CASTLING_ROOK_MAPS[castlingIndex] &
                 _boardFetcher.BitBoard(movingColor * BIT_BOARDS_PER_COLOR + ROOK_INDEX)) !=
                0 &&
                (CASTLING_SENSITIVE_FIELDS[castlingIndex] & blockedFigMap) == 0 &&
                (CASTLING_TOUCHED_FIELDS[castlingIndex] & fullMap) == 0) {
                auto castlings = _boardFetcher.Castlings();
                SetBitBoardBit(castlings, movingColor * CASTLINGS_PER_COLOR + KING_CASTLING_INDEX,
                               false);
                SetBitBoardBit(castlings, movingColor * CASTLINGS_PER_COLOR + QUEEN_CASTLING_INDEX,
                               false);

                cuda_Move mv{};

                // preparing basic move info
                mv.SetStartField(ExtractMsbPos(DEFAULT_KING_BOARDS[movingColor]));
                mv.SetStartBoardIndex(movingColor * BIT_BOARDS_PER_COLOR + KING_INDEX);
                mv.SetTargetField(CASTLING_NEW_KING_POS[castlingIndex]);
                mv.SetTargetBoardIndex(movingColor * BIT_BOARDS_PER_COLOR + KING_INDEX);
                mv.SetKilledBoardIndex(movingColor * BIT_BOARDS_PER_COLOR + ROOK_INDEX);
                mv.SetKilledFigureField(ExtractMsbPos(CASTLING_ROOK_MAPS[castlingIndex]));
                mv.SetElPassantField(INVALID_EL_PASSANT_FIELD);
                mv.SetCastlingRights(castlings);
                mv.SetCastlingType(1 + castlingIndex); // sentinel value
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
