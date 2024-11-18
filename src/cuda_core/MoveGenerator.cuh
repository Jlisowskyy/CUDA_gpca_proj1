//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef MOVEGENERATOR_CUH
#define MOVEGENERATOR_CUH


#include "ChessMechanics.cuh"
#include "cuda_BitOperations.cuh"
#include "Move.cuh"

#include "BishopMap.cuh"
#include "BlackPawnMap.cuh"
#include "WhitePawnMap.cuh"
#include "KingMap.cuh"
#include "KnightMap.cuh"
#include "QueenMap.cuh"
#include "RookMap.cuh"
#include "cuda_Array.cuh"

#include <array>
#include <map>
#include <queue>
#include <cassert>
#include <type_traits>

__device__ static constexpr uint16_t PromoFlags[]{
        0, KnightFlag, BishopFlag, RookFlag, QueenFlag,
};

__device__ static constexpr __uint64_t CASTLING_PSEUDO_LEGAL_BLOCKED = 0;

struct MoveGenerator : ChessMechanics {

    struct payload {
        cuda_Array<cuda_Move, 256> data{};
        size_t size = 0;

        [[nodiscard]] FAST_CALL const cuda_Move &operator[](size_t index) const {
            return data[index];
        }

        [[nodiscard]] FAST_CALL cuda_Move &operator[](size_t index) {
            return data[index];
        }

        FAST_CALL void Push(cuda_Move mv) {
            data[size++] = mv;
        }
    };

    // ------------------------------
    // Class Creation
    // ------------------------------

    MoveGenerator() = delete;

    FAST_CALL explicit MoveGenerator(const cuda_Board &bd) : ChessMechanics(bd) {}

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
        const auto [blockedFigMap, checksCount, checkType] = GetBlockedFieldBitMap(fullMap);

        assert(blockedFigMap != 0 && "Blocked fig map must at least contains fields controlled by king!");
        assert(
                checksCount <= 2 &&
                "We consider only 3 states: no-check, single-check, double-check -> invalid result was returned!"
        );

        payload results{};

        // depending on amount of checks branch to desired reaction
        switch (checksCount) {
            case 0:
                _noCheckGen<GenOnlyTacticalMoves>(results, fullMap, blockedFigMap);
                break;
            case 1:
                _singleCheckGen<GenOnlyTacticalMoves>(results, fullMap, blockedFigMap, checkType);
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

    template<bool GenOnlyTacticalMoves = false>
    FAST_DCALL payload GetMovesSlow() {
        cuda_Board bd = _board;

        payload load =
                GetPseudoLegalMoves<GenOnlyTacticalMoves>();

        payload loadRV{};
        for (size_t i = 0; i < load.size; ++i)
            if (MoveGenerator::IsLegal(bd, load.data[i]))
                loadRV.Push(load.data[i]);

        return loadRV;
    }


    template<bool GenOnlyTacticalMoves = false>
    FAST_CALL payload GetPseudoLegalMoves() {
        // allocate results container
        payload results{};

        // Generate bitboards with corresponding colors
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_board.MovingColor));
        const __uint64_t allyMap = GetColBitMap(_board.MovingColor);

        _processFigPseudoMoves<GenOnlyTacticalMoves, KnightMap>(
                results, enemyMap, allyMap
        );

        _processFigPseudoMoves<GenOnlyTacticalMoves, BishopMap>(
                results, enemyMap, allyMap
        );

        _processFigPseudoMoves<GenOnlyTacticalMoves, RookMap, true>(
                results, enemyMap, allyMap
        );

        _processFigPseudoMoves<GenOnlyTacticalMoves, QueenMap>(
                results, enemyMap, allyMap
        );

        if (_board.MovingColor == WHITE)
            _processPawnPseudoMoves<GenOnlyTacticalMoves, WhitePawnMap>(
                    results, enemyMap, allyMap
            );
        else
            _processPawnPseudoMoves<GenOnlyTacticalMoves, BlackPawnMap>(
                    results, enemyMap, allyMap
            );

        _processPlainKingMoves<GenOnlyTacticalMoves, true>(
                results, KING_NO_BLOCKED_MAP, allyMap, enemyMap
        );

        if constexpr (!GenOnlyTacticalMoves)
            _processKingCastlings<true>(
                    results, CASTLING_PSEUDO_LEGAL_BLOCKED, enemyMap | allyMap
            );

        return results;
    }

    [[nodiscard]] FAST_DCALL static bool IsLegal(cuda_Board &bd, cuda_Move mv) {
        // TODO: Possibility to improve performance by introducing global state holding pinned figures
        // TODO: and distinguish given states:
        // TODO: Castling -> (check for attacks on fields),
        // TODO: King Move / El Passant -> (check if king is attacked from king's perspective),
        // TODO: Rest of moves allowed only on line between pinned and king if pinned

        return mv.GetPackedMove().IsCastling() ?
               MoveGenerator::_isCastlingLegal(bd, mv) :
               MoveGenerator::_isNormalMoveLegal(bd, mv);
    }

    [[nodiscard]] __device__ __uint64_t CountMoves(cuda_Board &bd, int depth) {
        if (depth == 0)
            return 1;

        MoveGenerator mgen{bd};
        const auto moves = mgen.GetMovesSlow<false>();

        if (depth == 1) {
            return moves.size;
        }

        __uint64_t sum{};

        VolatileBoardData data{bd};
        for (size_t i = 0; i < moves.size; ++i) {
            cuda_Move::MakeMove(moves[i], bd);
            sum += CountMoves(bd, depth - 1);
            cuda_Move::UnmakeMove(moves[i], bd, data);
        }

        return sum;
    }

    using ChessMechanics::IsCheck;

    // ------------------------------
    // private methods
    // ------------------------------

private:

    [[nodiscard]] FAST_DCALL static bool _isCastlingLegal(cuda_Board &bd, cuda_Move mv) {
        ChessMechanics mech(bd);
        const auto [blocked, unused, unused1] =
                mech.GetBlockedFieldBitMap(mech.GetFullBitMap());

        // Refer to castling generating function in MoveGenerator, there is sentinel on index 0 somehow

        // Check if the king is not attacked and castling sensitive fields too
        return (CastlingSensitiveFields[mv.GetCastlingType() - 1] & blocked) == 0 &&
               (blocked & bd.GetFigBoard(bd.MovingColor, kingIndex)) == 0;
    }

    [[nodiscard]] FAST_DCALL static bool _isNormalMoveLegal(cuda_Board &bd, cuda_Move mv) {
        bd.BitBoards[mv.GetStartBoardIndex()] ^= cuda_MaxMsbPossible >> mv.GetStartField();
        bd.BitBoards[mv.GetTargetBoardIndex()] |= cuda_MaxMsbPossible >> mv.GetTargetField();
        bd.BitBoards[mv.GetKilledBoardIndex()] ^= cuda_MaxMsbPossible >> mv.GetKilledFigureField();

        auto reverse = [&]() {
            bd.BitBoards[mv.GetStartBoardIndex()] |= cuda_MaxMsbPossible >> mv.GetStartField();
            bd.BitBoards[mv.GetTargetBoardIndex()] ^= cuda_MaxMsbPossible >> mv.GetTargetField();
            bd.BitBoards[mv.GetKilledBoardIndex()] |= cuda_MaxMsbPossible >> mv.GetKilledFigureField();
        };

        ChessMechanics mechanics(bd);

        const __uint32_t enemyCol = SwapColor(bd.MovingColor);
        const __uint32_t kingsMsb = bd.GetKingMsbPos(bd.MovingColor);
        const __uint64_t fullBoard = mechanics.GetFullBitMap();

        // Checking rook's perspective
        const __uint64_t enemyRooks = bd.GetFigBoard(enemyCol, rooksIndex);
        const __uint64_t enemyQueens = bd.GetFigBoard(enemyCol, queensIndex);

        const __uint64_t kingsRookPerspective = RookMap::GetMoves(kingsMsb, fullBoard);

        if ((kingsRookPerspective & (enemyRooks | enemyQueens)) != 0)
            return (reverse(), false);

        // Checking bishop's perspective
        const __uint64_t enemyBishops = bd.GetFigBoard(enemyCol, bishopsIndex);
        const __uint64_t kingsBishopPerspective = BishopMap::GetMoves(kingsMsb, fullBoard);

        if ((kingsBishopPerspective & (enemyBishops | enemyQueens)) != 0)
            return (reverse(), false);

        // checking knights attacks
        const __uint64_t enemyKnights = bd.GetFigBoard(enemyCol, knightsIndex);
        const __uint64_t knightsPerspective = KnightMap::GetMoves(kingsMsb);

        if ((knightsPerspective & (enemyKnights)) != 0)
            return (reverse(), false);

        // pawns checks
        const __uint64_t enemyPawns = bd.GetFigBoard(enemyCol, pawnsIndex);
        const __uint64_t pawnAttacks =
                enemyCol == WHITE ? WhitePawnMap::GetAttackFields(enemyPawns) : BlackPawnMap::GetAttackFields(
                        enemyPawns);

        if ((pawnAttacks & (cuda_MaxMsbPossible >> kingsMsb)) != 0)
            return (reverse(), false);

        return (reverse(), true);
    }

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
    FAST_CALL void _singleCheckGen(payload &results, __uint64_t fullMap, __uint64_t blockedFigMap, int checkType) {
        assert(fullMap != 0 && "Full map is empty!");
        assert(checkType == simpleFigCheck || checkType == slidingFigCheck && "Invalid check type!");

        static constexpr __uint64_t UNUSED = 0;

        // simplifying figure search by distinguishing check types
        const auto [pinnedFigsMap, allowedTilesMap] = [&]() -> std::pair<__uint64_t, __uint64_t> {
            if (checkType == slidingFigCheck)
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
    FAST_CALL void _doubleCheckGen(payload &results, __uint64_t blockedFigMap) const {
        const __uint64_t allyMap = GetColBitMap(_board.MovingColor);
        const __uint64_t enemyMap = GetColBitMap(SwapColor(_board.MovingColor));
        _processPlainKingMoves<GenOnlyTacticalMoves>(results, blockedFigMap, allyMap, enemyMap);
    }

    template<bool GenOnlyTacticalMoves, class MapT, bool isCheck = false>
    FAST_CALL void _processPawnMoves(
            payload &results, __uint64_t enemyMap, __uint64_t allyMap, __uint64_t pinnedFigMap,
            [[maybe_unused]] __uint64_t allowedMoveFilter = 0
    ) {
        const __uint64_t promotingPawns = _board.BitBoards[MapT::GetBoardIndex(0)] & MapT::PromotingMask;
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

    template<bool GenOnlyTacticalMoves, class MapT>
    FAST_CALL void _processPawnPseudoMoves(
            payload &results, __uint64_t enemyMap, __uint64_t allyMap
    ) {
        __uint64_t mask{};

        if constexpr (std::is_same<MapT, WhitePawnMap>::value) {
            mask = WhitePawnMapConstants::PromotingMask;
        } else if constexpr (std::is_same<MapT, BlackPawnMap>::value) {
            mask = BlackPawnMapConstants::PromotingMask;
        } else {
            assert(false && "Invalid pawn map type detected!");
        }

        // Distinguish pawns that should be promoted from usual ones
        const __uint64_t promotingPawns = _board.BitBoards[MapT::GetBoardIndex(0)] & mask;
        const __uint64_t nonPromotingPawns = _board.BitBoards[MapT::GetBoardIndex(0)] ^ promotingPawns;
        const __uint64_t fullMap = enemyMap | allyMap;

        _processPawnAttackPseudoMoves<GenOnlyTacticalMoves, MapT>(
                results, enemyMap, fullMap, promotingPawns, nonPromotingPawns
        );

        _processPawnPlainPseudoMoves<GenOnlyTacticalMoves, MapT>(
                results, fullMap, promotingPawns, nonPromotingPawns
        );

        _processElPassantPseudoMoves<MapT>(results, nonPromotingPawns);
    }

    template<bool IsQSearch, class MapT>
    __device__ void _processPawnAttackPseudoMoves(
            payload &results, __uint64_t enemyMap, __uint64_t fullMap, __uint64_t promoMoves, __uint64_t nonPromoMoves
    ) {
        // NOTE: attack can overlap each other so attacking moves should be generated separately for each piece

        // iterate through every non promoting piece
        while (nonPromoMoves) {
            const int pawnPos = ExtractMsbPos(nonPromoMoves);
            const __uint64_t pawnBitBoard = cuda_MaxMsbPossible >> pawnPos;

            // generate attacking moves
            __uint64_t attackPseudoMoves = MapT::GetAttackFields(pawnBitBoard) & enemyMap;

            while (attackPseudoMoves) {
                const int pawnTarget = ExtractMsbPos(attackPseudoMoves);
                const __uint64_t pawnTargetBitBoard = cuda_MaxMsbPossible >> pawnTarget;
                const size_t attackedFigBoardIndex = GetIndexOfContainingBitBoard(pawnTargetBitBoard,
                                                                                  SwapColor(_board.MovingColor));

                cuda_Move mv{};

                // preparing basic move info
                mv.SetStartField(pawnPos);
                mv.SetStartBoardIndex(MapT::GetBoardIndex(0));
                mv.SetTargetField(pawnTarget);
                mv.SetTargetBoardIndex(MapT::GetBoardIndex(0));
                mv.SetKilledBoardIndex(attackedFigBoardIndex);
                mv.SetKilledFigureField(pawnTarget);
                mv.SetElPassantField(InvalidElPassantField);
                mv.SetCasltingRights(_board.Castlings);
                mv.SetMoveType(CaptureFlag);

                if (_isPawnGivingCheck<MapT>(pawnTargetBitBoard))
                    mv.SetCheckType();

                results.Push(mv);
                attackPseudoMoves ^= pawnTargetBitBoard;
            }

            nonPromoMoves ^= pawnBitBoard;
        }

        // iterate through every promoting piece
        while (promoMoves) {
            const int pawnPos = ExtractMsbPos(promoMoves);
            const __uint64_t pawnBitBoard = cuda_MaxMsbPossible >> pawnPos;

            // generate attacking moves
            __uint64_t attackPseudoMoves = MapT::GetAttackFields(pawnBitBoard) & enemyMap;

            while (attackPseudoMoves) {
                static constexpr size_t startInd = queensIndex;
                static constexpr size_t limitInd = queensIndex - 1;

                const int pawnTarget = ExtractMsbPos(attackPseudoMoves);
                const __uint64_t pawnTargetBitBoard = cuda_MaxMsbPossible >> pawnTarget;
                const size_t attackedFigBoardIndex = GetIndexOfContainingBitBoard(pawnTargetBitBoard,
                                                                                  SwapColor(_board.MovingColor));

                // iterating through upgradable pieces
                for (size_t i = startInd; i > limitInd; --i) {
                    const auto targetBoard = _board.MovingColor * BitBoardsPerCol + i;

                    cuda_Move mv{};

                    // preparing basic move info
                    mv.SetStartField(pawnPos);
                    mv.SetStartBoardIndex(MapT::GetBoardIndex(0));
                    mv.SetTargetField(pawnTarget);
                    mv.SetTargetBoardIndex(targetBoard);
                    mv.SetKilledBoardIndex(attackedFigBoardIndex);
                    mv.SetKilledFigureField(pawnTarget);
                    mv.SetElPassantField(InvalidElPassantField);
                    mv.SetCasltingRights(_board.Castlings);
                    mv.SetMoveType(CaptureFlag | PromoFlag | PromoFlags[i]);

                    if (_isPromotingPawnGivingCheck<MapT>(pawnTarget, fullMap ^ pawnBitBoard, targetBoard))
                        mv.SetCheckType();

                    results.Push(mv);
                }

                attackPseudoMoves ^= pawnTargetBitBoard;
            }

            promoMoves ^= pawnBitBoard;
        }
    }

    template<bool GenOnlyTacticalMoves, class MapT>
    __device__ void _processPawnPlainPseudoMoves(
            payload &results, __uint64_t fullMap, __uint64_t promoPieces, __uint64_t nonPromoPieces
    ) {
        // iterate through every promoting piece
        while (promoPieces) {
            const int pawnPos = ExtractMsbPos(promoPieces);
            const __uint64_t pawnBitBoard = cuda_MaxMsbPossible >> pawnPos;

            // generate plain moves
            __uint64_t plainPseudoMoves = MapT::GetSinglePlainMoves(pawnBitBoard, fullMap);

            while (plainPseudoMoves) {
                static constexpr size_t startInd = queensIndex;
                static constexpr size_t limitInd = queensIndex - 1;

                const int pawnTarget = ExtractMsbPos(plainPseudoMoves);
                const __uint64_t pawnTargetBitBoard = cuda_MaxMsbPossible >> pawnTarget;

                // iterating through upgradable pieces
                for (size_t i = startInd; i > limitInd; --i) {
                    const auto targetBoard = _board.MovingColor * BitBoardsPerCol + i;

                    cuda_Move mv{};

                    // preparing basic move info
                    mv.SetStartField(pawnPos);
                    mv.SetStartBoardIndex(MapT::GetBoardIndex(0));
                    mv.SetTargetField(pawnTarget);
                    mv.SetTargetBoardIndex(targetBoard);
                    mv.SetKilledBoardIndex(SentinelBoardIndex);
                    mv.SetElPassantField(InvalidElPassantField);
                    mv.SetCasltingRights(_board.Castlings);
                    mv.SetMoveType(PromoFlag | PromoFlags[i]);

                    if (_isPromotingPawnGivingCheck<MapT>(pawnTarget, fullMap ^ pawnBitBoard, targetBoard))
                        mv.SetCheckType();

                    results.Push(mv);
                }

                plainPseudoMoves ^= pawnTargetBitBoard;
            }

            promoPieces ^= pawnBitBoard;
        }

        // All tactical moves generated
        if constexpr (GenOnlyTacticalMoves)
            return;

        __uint64_t startMask{};

        if constexpr (std::is_same<MapT, WhitePawnMap>::value) {
            startMask = WhitePawnMapConstants::StartMask;
        } else if constexpr (std::is_same<MapT, BlackPawnMap>::value) {
            startMask = BlackPawnMapConstants::StartMask;
        } else {
            assert(false && "Invalid pawn map type detected!");
        }

        __uint64_t firstMoves = MapT::GetSinglePlainMoves(nonPromoPieces, fullMap);
        __uint64_t firstPawns = MapT::RevertSinglePlainMoves(firstMoves);
        __uint64_t secondMoves = MapT::GetSinglePlainMoves(firstMoves & MapT::GetSinglePlainMoves(startMask, fullMap),
                                                           fullMap);
        __uint64_t secondPawns = MapT::RevertSinglePlainMoves(MapT::RevertSinglePlainMoves(secondMoves));

        // go through single upfront moves
        while (firstMoves) {
            const int moveMsb = ExtractMsbPos(firstMoves);
            const int pawnMsb = ExtractMsbPos(firstPawns);
            const __uint64_t pawnBitBoard = cuda_MaxMsbPossible >> pawnMsb;
            const __uint64_t moveBitBoard = cuda_MaxMsbPossible >> moveMsb;

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(pawnMsb);
            mv.SetStartBoardIndex(MapT::GetBoardIndex(0));
            mv.SetTargetField(moveMsb);
            mv.SetTargetBoardIndex(MapT::GetBoardIndex(0));
            mv.SetKilledBoardIndex(SentinelBoardIndex);
            mv.SetElPassantField(InvalidElPassantField);
            mv.SetCasltingRights(_board.Castlings);

            if (_isPawnGivingCheck<MapT>(moveBitBoard))
                mv.SetCheckType();

            results.Push(mv);
            firstMoves ^= moveBitBoard;
            firstPawns ^= pawnBitBoard;
        }

        // go through double upfront moves
        while (secondMoves) {
            const int moveMsb = ExtractMsbPos(secondMoves);
            const int pawnMsb = ExtractMsbPos(secondPawns);
            const __uint64_t pawnBitBoard = cuda_MaxMsbPossible >> pawnMsb;
            const __uint64_t moveBitBoard = cuda_MaxMsbPossible >> moveMsb;

            cuda_Move mv{};

            // preparing basic move info
            mv.SetStartField(pawnMsb);
            mv.SetStartBoardIndex(MapT::GetBoardIndex(0));
            mv.SetTargetField(moveMsb);
            mv.SetTargetBoardIndex(MapT::GetBoardIndex(0));
            mv.SetKilledBoardIndex(SentinelBoardIndex);
            mv.SetElPassantField(moveMsb);
            mv.SetCasltingRights(_board.Castlings);

            if (_isPawnGivingCheck<MapT>(moveBitBoard))
                mv.SetCheckType();

            results.Push(mv);
            secondMoves ^= moveBitBoard;
            secondPawns ^= pawnBitBoard;
        }
    }

    template<class MapT>
    FAST_DCALL void _processElPassantPseudoMoves(
            payload &results, __uint64_t pieces
    ) {
        if (_board.ElPassantField == InvalidElPassantBitBoard)
            return;

        const __uint64_t suspectedFields = MapT::GetElPassantSuspectedFields(_board.ElPassantField);
        __uint64_t elPassantPawns = pieces & suspectedFields;

        while (elPassantPawns) {
            const int pawnMsb = ExtractMsbPos(elPassantPawns);
            const __uint64_t pawnMap = cuda_MaxMsbPossible >> pawnMsb;
            const __uint64_t moveBitMap = MapT::GetElPassantMoveField(_board.ElPassantField);

            // preparing basic move information
            cuda_Move mv{};
            mv.SetStartField(pawnMsb);
            mv.SetStartBoardIndex(MapT::GetBoardIndex(0));
            mv.SetTargetField(ExtractMsbPos(moveBitMap));
            mv.SetTargetBoardIndex(MapT::GetBoardIndex(0));
            mv.SetKilledBoardIndex(MapT::GetEnemyPawnBoardIndex());
            mv.SetKilledFigureField(ExtractMsbPos(_board.ElPassantField));
            mv.SetElPassantField(InvalidElPassantField);
            mv.SetCasltingRights(_board.Castlings);
            mv.SetMoveType(CaptureFlag);

            if (_isPawnGivingCheck<MapT>(moveBitMap))
                mv.SetCheckType();

            results.Push(mv);
            elPassantPawns ^= pawnMap;
        }
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

        while (possiblePawnsToMove) {
            const __uint64_t pawnMap = cuda_MaxMsbPossible >> ExtractMsbPos(possiblePawnsToMove);

            // checking whether move would affect horizontal line attacks on king
            const __uint64_t processedPawns = pawnMap | _board.ElPassantField;
            const __uint64_t cleanedFromPawnsMap = fullMap ^ processedPawns;
            if (const __uint64_t kingHorizontalLine =
                        RookMap::GetMoves(_board.GetKingMsbPos(_board.MovingColor), cleanedFromPawnsMap) &
                        MapT::EnemyElPassantMask;
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


            results.Push(mv);
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

    template<
            bool GenOnlyTacticalMoves, class MapT, bool checkForCastling = false
    >
    FAST_DCALL void _processFigPseudoMoves(
            payload &results, __uint64_t enemyMap, __uint64_t allyMap
    ) {
        assert(enemyMap != 0 && "Enemy map is empty!");
        assert(allyMap != 0 && "Ally map is empty!");

        // prepare full map and extract figures BitMap from the board
        const __uint64_t fullMap = enemyMap | allyMap;
        __uint64_t figures = _board.BitBoards[MapT::GetBoardIndex(_board.MovingColor)];

        // processing unpinned moves
        while (figures) {
            // processing moves
            const int figPos = ExtractMsbPos(figures);
            const __uint64_t figBoard = cuda_MaxMsbPossible >> figPos;

            // Generating actual pseudo legal moves
            const __uint64_t figMoves = MapT::GetMoves(figPos, fullMap, enemyMap) & ~allyMap;

            // Performing checks for castlings
            __uint32_t updatedCastlings = _board.Castlings;
            if constexpr (checkForCastling)
                SetBitBoardBit(updatedCastlings, RookMap::GetMatchingCastlingIndex(_board, figBoard), false);

            // preparing moves
            const __uint64_t attackMoves = figMoves & enemyMap;
            [[maybe_unused]] const __uint64_t nonAttackingMoves = figMoves ^ attackMoves;

            // processing move consequences
            if constexpr (!GenOnlyTacticalMoves)
                // Don't add simple moves when we should generate only attacking moves
                _processNonAttackingMoves<MapT, false>(
                        results, nonAttackingMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard,
                        updatedCastlings, fullMap
                );

            _processAttackingMoves<MapT, GenOnlyTacticalMoves, false>(
                    results, attackMoves, MapT::GetBoardIndex(_board.MovingColor), figBoard, updatedCastlings,
                    fullMap
            );

            figures ^= figBoard;
        }
    }


    // TODO: improve readability of code below
    template<
            class MapT, bool promotePawns,
            __uint64_t (*elPassantFieldDeducer)(__uint64_t, __uint64_t) = nullptr>
    __device__ void _processNonAttackingMoves(
            payload &results, __uint64_t nonAttackingMoves, size_t figBoardIndex, __uint64_t startField,
            __uint32_t castlings, __uint64_t fullMap
    ) const {
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

                results.Push(mv);
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

                    results.Push(mv);
                }
            }

            nonAttackingMoves ^= moveBoard;
        }
    }

    template<class MapT, bool IsQsearch, bool promotePawns>
    __device__ void _processAttackingMoves(
            payload &results, __uint64_t attackingMoves, size_t figBoardIndex, __uint64_t startField,
            __uint32_t castlings, __uint64_t fullMap
    ) const  {
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

                results.Push(mv);
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

                    results.Push(mv);
                }
            }

            attackingMoves ^= moveBoard;
        }
    }

    static constexpr __uint64_t KING_NO_BLOCKED_MAP = (~static_cast<__uint64_t>(0));

    // TODO: test copying all old castlings
    template<bool GenOnlyTacticalMoves, bool GeneratePseudoMoves = false>
    __device__ void
    _processPlainKingMoves(payload &results, __uint64_t blockedFigMap, __uint64_t allyMap, __uint64_t enemyMap) const {
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

                results.Push(mv);

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

            results.Push(mv);

            attackingMoves ^= newKingBoard;
        }
    }



    // TODO: simplify ifs??
    // TODO: cleanup left castling available when rook is dead then propagate no castling checking?
    template<bool GeneratePseudoLegalMoves = false>
    FAST_DCALL void _processKingCastlings(payload &results, __uint64_t blockedFigMap, __uint64_t fullMap) const {
        assert(fullMap != 0 && "Full map is empty!");

        for (size_t i = 0; i < CastlingsPerColor; ++i)
            if (const size_t castlingIndex = _board.MovingColor * CastlingsPerColor + i;
                    _board.GetCastlingRight(castlingIndex) &&
                    (CastlingsRookMaps[castlingIndex] &
                     _board.BitBoards[_board.MovingColor * BitBoardsPerCol + rooksIndex]) != 0 &&
                    (CastlingSensitiveFields[castlingIndex] &
                     (GeneratePseudoLegalMoves ? CASTLING_PSEUDO_LEGAL_BLOCKED : blockedFigMap)) == 0 &&
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

                results.Push(mv);
            }
    }
};

#endif // MOVEGENERATOR_CUH
