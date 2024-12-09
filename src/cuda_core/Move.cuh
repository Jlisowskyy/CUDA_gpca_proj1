#ifndef CUDA_MOVE_CUH
#define CUDA_MOVE_CUH

#include <cuda_runtime.h>
#include <string>

#include "cuda_Board.cuh"
#include "cuda_PackedBoard.cuh"

/*      Class encodes chess move and heuristic evaluation
 *  together inside some uint64_t value. Encoding corresponds to value below:
 *  (bit indexed from lsb to msb manner)
 *  - bits 52-56 - encodes castling rights (4 bits) - "CastlingRights" - OBLIG IMPORTANT: should always be filled, in
 * case no changes was done simply copy previous values
 */

/*      Class encodes chess move and heuristic evaluation
 *  together inside some uint64_t value. Encoding corresponds to value below:
 *  - bits  0-15 - encodes simplistic heuristic evaluation of the move used inside inflight move sorting,
 *  - bits 16-31 - contains cuda_PackedMove instance
 *  - bits 32-35 - encodes board index from which figure moved
 *  - bits 36-39 - encodes board index to which figure moved - differs from StartBoardIndex only in case of promotion
 *  - bits 40-43 - encodes board index on which figure was killed used in case of attacking move
 *  - bits 44-46 - encodes type of performed castling e.g. white king-side castling
 *  - bit  47    - indicates whether given moves checks enemy king
 *  - bits 48-53 - encodes field on which figure was killed - differs from TargetField only in case of el passant
 *  - bits 54-59 - encodes new elPassant field
 *  - bits 60-63 - encodes new castling rights
 */

/*      IMPORTANT NOTE:
 *  ALL SET METHODS WORK CORRECTLY ONLY
 *  ON BY DEFAULT INITIALIZED OBJECTS EVERY ONE OF THEM WORKS ONCE
 */

__device__ __constant__ static constexpr uint16_t PromoFlag = 0b1000;
__device__ __constant__ static constexpr uint16_t CaptureFlag = 0b100;
__device__ __constant__ static constexpr uint16_t CastlingFlag = 0b10;
static constexpr uint16_t CastlingFlagCPU = 0b10;
__device__ __constant__ static constexpr uint16_t QueenFlag = 0;
__device__ __constant__ static constexpr uint16_t RookFlag = 0b1;
__device__ __constant__ static constexpr uint16_t BishopFlag = 0b10;
__device__ __constant__ static constexpr uint16_t KnightFlag = 0b11;
__device__ __constant__ static constexpr uint16_t PromoSpecBits = 0b11;

__device__ __constant__ static constexpr uint16_t MoveTypeBits = 0xF << 12;
static constexpr uint16_t MoveTypeBitsCPU = 0xF << 12;
__device__ __constant__ static constexpr uint16_t Bit6 = 0b111111;
__device__ __constant__ static constexpr uint16_t Bit4 = 0b1111;
__device__ __constant__ static constexpr uint16_t Bit3 = 0b111;
__device__ __constant__ static constexpr uint16_t PromoBit = PromoFlag << 12;
__device__ __constant__ static constexpr uint16_t CaptureBit = CaptureFlag << 12;
__device__ __constant__ static constexpr uint16_t CastlingBits = CastlingFlag << 12;
static constexpr uint16_t CastlingBitsCPU = CastlingFlagCPU << 12;

class cuda_Move;

FAST_CALL_ALWAYS std::pair<char, char> ConvertToCharPos(const int boardPosMsb) {
    const uint32_t boardPos = ConvertToReversedPos(boardPosMsb);
    return {static_cast<char>('a' + (boardPos % 8)), static_cast<char>('1' + (boardPos / 8))};
}

#ifdef NDEBUG

#define ASSERT_EVAL(a, b)
#define ASSERT_EVAL_DEV(a, b, c)

#else

#define ASSERT_EVAL_DEV(expected, actual, mv)  \
{ \
    if (expected != actual) { \
        printLock(); \
        printf("Expected: %d, Actual: %d on move: %u\n", expected, actual, mv); \
        assert(false); \
    }  \
}
#define ASSERT_EVAL(expected, actual)  \
{ \
    if (expected != actual) { \
        printf("Expected: %d, Actual: %d\n", expected, actual); \
        assert(false); \
    }  \
}
#endif

struct cuda_PackedMove final {
    // ------------------------------
    // Class creation
    // ------------------------------

    FAST_CALL explicit cuda_PackedMove(const uint16_t packedMove) : _packedMove(packedMove) {
    }

    cuda_PackedMove() = default;

    ~cuda_PackedMove() = default;

    cuda_PackedMove(const cuda_PackedMove &other) = default;

    cuda_PackedMove &operator=(const cuda_PackedMove &other) = default;

    cuda_PackedMove(cuda_PackedMove &&other) = default;

    cuda_PackedMove &operator=(cuda_PackedMove &&other) = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] std::string GetLongAlgebraicNotationCPU() const {
        static constexpr char PromoFigs[] = {'q', 'r', 'b', 'n'};
        std::string rv;

        auto [c1, c2] = ConvertToCharPos((int) GetStartFieldCPU());
        rv += c1;
        rv += c2;
        auto [c3, c4] = ConvertToCharPos((int) GetTargetFieldCPU());
        rv += c3;
        rv += c4;

        if (IsPromo()) {
            rv += PromoFigs[GetMoveType() & PromoSpecBits];
        }

        return rv;
    }

    FAST_CALL friend bool operator==(const cuda_PackedMove a, const cuda_PackedMove b) {
        return a._packedMove == b._packedMove;
    }

    FAST_CALL friend bool operator!=(const cuda_PackedMove a, const cuda_PackedMove b) { return !(a == b); }

    FAST_DCALL_ALWAYS void SetStartField(const uint16_t startField) { _packedMove |= startField; }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetStartField() const {
        __constant__ static constexpr uint16_t StartFieldMask = Bit6;
        return StartFieldMask & _packedMove;
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetStartFieldCPU() const {
        static constexpr uint16_t StartFieldMask = Bit6;
        return StartFieldMask & _packedMove;
    }

    FAST_DCALL_ALWAYS void SetTargetField(const uint16_t targetField) { _packedMove |= targetField << 6; }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetTargetField() const {
        __constant__ static constexpr uint16_t TargetFieldMask = Bit6 << 6;
        return (_packedMove & TargetFieldMask) >> 6;
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetTargetFieldCPU() const {
        static constexpr uint16_t TargetFieldMask = Bit6 << 6;
        return (_packedMove & TargetFieldMask) >> 6;
    }

    [[nodiscard]] FAST_CALL_ALWAYS bool IsEmpty() const { return _packedMove == 0; }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsQuiet() const { return (_packedMove & MoveTypeBits) == 0; }

    [[nodiscard]] FAST_CALL_ALWAYS bool IsCapture() const { return (_packedMove & CaptureBit) != 0; }

    [[nodiscard]] FAST_CALL_ALWAYS bool IsPromo() const { return (_packedMove & PromoBit) != 0; }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsCastling() const { return (_packedMove & MoveTypeBits) == CastlingBits; }

    [[nodiscard]] bool IsCastlingCPU() const { return (_packedMove & MoveTypeBitsCPU) == CastlingBitsCPU; }

    FAST_DCALL_ALWAYS void SetMoveType(const uint16_t MoveType) { _packedMove |= MoveType << 12; }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetMoveType() const { return (_packedMove & MoveTypeBits) >> 12; }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsValidMove() const { return !IsEmpty(); }

    // debugging tool
    [[nodiscard]] FAST_DCALL_ALWAYS bool IsOkayMove() const {
        return !IsEmpty() && GetTargetField() != GetStartField();
    }

    [[nodiscard]] bool IsOkayMoveCPU() const {
        return !IsEmpty() && GetTargetFieldCPU() != GetStartFieldCPU();
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t Dump() const { return _packedMove; }

    // ------------------------------
    // Class fields
    // ------------------------------

private:
    friend cuda_Move;

    uint16_t _packedMove{};
};

/* Class used to preserve some crucial board state between moves */
struct VolatileBoardData {
    using _fetcher_t = cuda_PackedBoard<PACKED_BOARD_DEFAULT_SIZE>::BoardFetcher;

    VolatileBoardData() = delete;

    FAST_CALL explicit VolatileBoardData(const cuda_Board &bd)
        : Castlings(bd.Castlings), OldElPassant(bd.ElPassantField) {
    }

    FAST_CALL explicit VolatileBoardData(uint32_t c, uint64_t ep)
        : Castlings(c), OldElPassant(ep) {
    }

    FAST_CALL explicit VolatileBoardData(const _fetcher_t &fetcher)
        : Castlings(fetcher.Castlings()), OldElPassant(fetcher.ElPassantField()) {
    }

    const uint32_t Castlings;
    const uint64_t OldElPassant;
};

__device__ __constant__ static constexpr uint32_t move_CastlingIdxArr[5]{
    SENTINEL_BOARD_INDEX, W_ROOK_INDEX, W_ROOK_INDEX, B_ROOK_INDEX, B_ROOK_INDEX
};

__device__ __constant__ static constexpr uint64_t move_CastlingNewKingPos[5]{
    1LLU, CASTLING_NEW_ROOK_MAPS[0], CASTLING_NEW_ROOK_MAPS[1],
    CASTLING_NEW_ROOK_MAPS[2], CASTLING_NEW_ROOK_MAPS[3]
};

static constexpr uint32_t move_CastlingIdxArrCPU[5]{
    SENTINEL_BOARD_INDEX, W_ROOK_INDEX, W_ROOK_INDEX, B_ROOK_INDEX, B_ROOK_INDEX
};

static constexpr uint64_t move_CastlingNewKingPosCPU[5]{
    1LLU, CASTLING_NEW_ROOK_MAPS[0], CASTLING_NEW_ROOK_MAPS[1],
    CASTLING_NEW_ROOK_MAPS[2], CASTLING_NEW_ROOK_MAPS[3]
};


class cuda_Move final {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    // This construction does not initialize crucial fields what must be done
    FAST_DCALL_ALWAYS explicit cuda_Move(const cuda_PackedMove mv) : _packedMove(mv) {
    }

    explicit cuda_Move(cpu::external_move eMove) : _packedMove(eMove[0]), _packedIndexes(eMove[1]),
                                                   _packedMisc(eMove[2]) {
    }

    cuda_Move() = default;

    ~cuda_Move() = default;

    cuda_Move(const cuda_Move &other) = default;

    cuda_Move &operator=(const cuda_Move &other) = default;

    cuda_Move(cuda_Move &&other) = default;

    cuda_Move &operator=(cuda_Move &&other) = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    FAST_DCALL_ALWAYS friend bool operator==(const cuda_Move a, const cuda_Move b) {
        return a._packedMove == b._packedMove;
    }

    FAST_DCALL_ALWAYS friend bool operator!=(const cuda_Move a, const cuda_Move b) { return !(a == b); }

    [[nodiscard]] FAST_CALL cuda_PackedMove GetPackedMove() const { return _packedMove; }

    FAST_DCALL_ALWAYS void SetMoveType(const uint16_t MoveType) { _packedMove.SetMoveType(MoveType); }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetMoveType() const { return _packedMove.GetMoveType(); }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsQuietMove() const {
        return !_packedMove.IsCapture() && !_packedMove.IsPromo();
    }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsValidMove() const { return _packedMove.IsValidMove(); }

    // debugging tool
    [[nodiscard]] FAST_DCALL_ALWAYS bool IsOkayMove() const { return _packedMove.IsOkayMove(); }

    [[nodiscard]] bool IsOkayMoveCPU() const { return _packedMove.IsOkayMoveCPU(); }

    template<uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
    FAST_DCALL_ALWAYS static void MakeMove(const cuda_Move mv, cuda_PackedBoard<NUM_BOARDS>::BoardFetcher fetcher) {
        assert(mv.IsOkayMove() && "Given move is not valid!");

        // removing the old piece from the board
        uint64_t prevBoard = fetcher.BitBoard(mv.GetStartBoardIndex());
        fetcher.SetBitBoard(prevBoard ^ (cuda_MaxMsbPossible >> mv.GetStartField()), mv.GetStartBoardIndex());

        // placing the figure on new field
        prevBoard = fetcher.BitBoard(mv.GetTargetBoardIndex());
        fetcher.SetBitBoard(prevBoard | (cuda_MaxMsbPossible >> mv.GetTargetField()), mv.GetTargetBoardIndex());

        // removing the killed figure in case no figure is killed index should be indicating to the sentinel
        prevBoard = fetcher.BitBoard(mv.GetKilledBoardIndex());
        fetcher.SetBitBoard(prevBoard ^ (cuda_MaxMsbPossible >> mv.GetKilledFigureField()), mv.GetKilledBoardIndex());

        // applying new castling rights
        fetcher.Castlings() = mv.GetCastlingRights();

        // applying new el passant field
        fetcher.SetElPassantField(cuda_MaxMsbPossible >> mv.GetElPassantField());

        // applying additional castling operation
        const uint32_t boardIndex = move_CastlingIdxArr[mv.GetCastlingType()];
        const uint64_t newKingPos = move_CastlingNewKingPos[mv.GetCastlingType()];
        fetcher.SetBitBoard(fetcher.BitBoard(boardIndex) | newKingPos, boardIndex);

        /* Update material value */
        fetcher.MaterialEval() -= mv.IsAttackingMove() * FIG_VALUES[mv.GetKilledBoardIndex()];
        fetcher.MaterialEval() += FIG_VALUES[mv.GetTargetBoardIndex()] - FIG_VALUES[mv.GetStartBoardIndex()];

        fetcher.ChangePlayingColor();

        ASSERT_EVAL_DEV(fetcher.EvaluateMaterial(), fetcher.MaterialEval(), mv.GetPackedMove().Dump());
    }

    static void MakeMove(const cuda_Move mv, cuda_Board &bd) {
        assert(mv.IsOkayMoveCPU() && "Given move is not valid!");

        // removing the old piece from the board
        bd.BitBoards[mv.GetStartBoardIndexCPU()] ^= cuda_MaxMsbPossible >> mv.GetStartFieldCPU();

        // placing the figure on new field
        bd.BitBoards[mv.GetTargetBoardIndexCPU()] |= cuda_MaxMsbPossible >> mv.GetTargetFieldCPU();

        // removing the killed figure in case no figure is killed index should be indicating to the sentinel
        bd.BitBoards[mv.GetKilledBoardIndexCPU()] ^= cuda_MaxMsbPossible >> mv.GetKilledFigureFieldCPU();

        // applying new castling rights
        bd.Castlings = mv.GetCastlingRightsCPU();

        // applying new el passant field
        bd.ElPassantField = cuda_MaxMsbPossible >> mv.GetElPassantFieldCPU();

        // applying additional castling operation
        const uint32_t boardIndex = move_CastlingIdxArrCPU[mv.GetCastlingTypeCPU()];
        const uint64_t newKingPos = move_CastlingNewKingPosCPU[mv.GetCastlingTypeCPU()];
        bd.BitBoards[boardIndex] |= newKingPos;

        bd.ChangePlayingColor();

        bd.MaterialEval -= mv.IsAttackingMove() * FIG_VALUES_CPU[mv.GetKilledBoardIndexCPU()];
        bd.MaterialEval += FIG_VALUES_CPU[mv.GetTargetBoardIndexCPU()] - FIG_VALUES_CPU[mv.GetStartBoardIndexCPU()];

        ASSERT_EVAL(bd.EvaluateMaterial(), bd.MaterialEval);
    }

    [[nodiscard]] FAST_CALL_ALWAYS bool IsAttackingMove() const { return _packedMove.IsCapture(); }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsEmpty() const { return _packedMove.IsEmpty(); }

    template<uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
    FAST_DCALL_ALWAYS static void UnmakeMove(const cuda_Move mv, cuda_PackedBoard<NUM_BOARDS>::BoardFetcher fetcher,
                                             const VolatileBoardData &data) {
        assert(mv.IsOkayMove() && "Given move is not valid!");

        fetcher.ChangePlayingColor();

        // placing the piece on old board
        uint64_t prevBoard = fetcher.BitBoard(mv.GetStartBoardIndex());
        fetcher.SetBitBoard(prevBoard | (cuda_MaxMsbPossible >> mv.GetStartField()), mv.GetStartBoardIndex());

        // removing the figure from the new field
        prevBoard = fetcher.BitBoard(mv.GetTargetBoardIndex());
        fetcher.SetBitBoard(prevBoard ^ (cuda_MaxMsbPossible >> mv.GetTargetField()), mv.GetTargetBoardIndex());

        // placing the killed figure in good place
        prevBoard = fetcher.BitBoard(mv.GetKilledBoardIndex());
        fetcher.SetBitBoard(prevBoard | (cuda_MaxMsbPossible >> mv.GetKilledFigureField()), mv.GetKilledBoardIndex());

        // recovering old castlings
        fetcher.Castlings() = data.Castlings;

        // recovering old el passant field
        fetcher.SetElPassantField(data.OldElPassant);

        // reverting castling operation
        const uint32_t boardIndex = move_CastlingIdxArr[mv.GetCastlingType()];
        const uint64_t newKingPos = move_CastlingNewKingPos[mv.GetCastlingType()];

        fetcher.SetBitBoard(fetcher.BitBoard(boardIndex) ^ newKingPos, boardIndex);

        /* Update material value */
        fetcher.MaterialEval() += mv.IsAttackingMove() * FIG_VALUES[mv.GetKilledBoardIndex()];
        fetcher.MaterialEval() -= FIG_VALUES[mv.GetTargetBoardIndex()] - FIG_VALUES[mv.GetStartBoardIndex()];


        ASSERT_EVAL_DEV(fetcher.EvaluateMaterial(), fetcher.MaterialEval(), mv.GetPackedMove().Dump());
    }

    static void UnmakeMove(const cuda_Move mv, cuda_Board &bd, const VolatileBoardData &data) {
        assert(mv.IsOkayMoveCPU() && "Given move is not valid!");

        bd.ChangePlayingColor();

        // placing the piece on old board
        bd.BitBoards[mv.GetStartBoardIndexCPU()] |= cuda_MaxMsbPossible >> mv.GetStartFieldCPU();

        // removing the figure from the new field
        bd.BitBoards[mv.GetTargetBoardIndexCPU()] ^= cuda_MaxMsbPossible >> mv.GetTargetFieldCPU();

        // placing the killed figure in good place
        bd.BitBoards[mv.GetKilledBoardIndexCPU()] |= cuda_MaxMsbPossible >> mv.GetKilledFigureFieldCPU();

        // recovering old castlings
        bd.Castlings = data.Castlings;

        // recovering old el passant field
        bd.ElPassantField = data.OldElPassant;

        // reverting castling operation
        const uint32_t boardIndex = move_CastlingIdxArrCPU[mv.GetCastlingTypeCPU()];
        const uint64_t newKingPos = move_CastlingNewKingPosCPU[mv.GetCastlingTypeCPU()];
        bd.BitBoards[boardIndex] ^= newKingPos;

        bd.MaterialEval += mv.IsAttackingMove() * FIG_VALUES_CPU[mv.GetKilledBoardIndexCPU()];
        bd.MaterialEval -= FIG_VALUES_CPU[mv.GetTargetBoardIndexCPU()] - FIG_VALUES_CPU[mv.GetStartBoardIndexCPU()];

        ASSERT_EVAL(bd.EvaluateMaterial(), bd.MaterialEval);
    }

    FAST_DCALL_ALWAYS void SetStartField(const uint16_t startField) { _packedMove.SetStartField(startField); }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetStartField() const { return _packedMove.GetStartField(); }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetStartFieldCPU() const { return _packedMove.GetStartFieldCPU(); }


    FAST_DCALL_ALWAYS void SetTargetField(const uint16_t targetField) { _packedMove.SetTargetField(targetField); }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetTargetField() const { return _packedMove.GetTargetField(); }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetTargetFieldCPU() const { return _packedMove.GetTargetFieldCPU(); }

    FAST_DCALL_ALWAYS void SetStartBoardIndex(const uint16_t startBoard) { _packedIndexes |= startBoard; }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetStartBoardIndex() const {
        __constant__ static constexpr uint16_t StartBoardMask = Bit4;

        return _packedIndexes & StartBoardMask;
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetStartBoardIndexCPU() const {
        static constexpr uint16_t StartBoardMask = Bit4;

        return _packedIndexes & StartBoardMask;
    }

    FAST_DCALL_ALWAYS void SetTargetBoardIndex(const uint16_t targetBoardIndex) {
        _packedIndexes |= targetBoardIndex << 4;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetTargetBoardIndex() const {
        __constant__ static constexpr uint16_t TargetBoardIndexMask = Bit4 << 4;

        return (_packedIndexes & TargetBoardIndexMask) >> 4;
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetTargetBoardIndexCPU() const {
        static constexpr uint16_t TargetBoardIndexMask = Bit4 << 4;

        return (_packedIndexes & TargetBoardIndexMask) >> 4;
    }

    FAST_DCALL_ALWAYS void SetKilledBoardIndex(const uint16_t killedBoardIndex) {
        _packedIndexes |= killedBoardIndex << 8;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetKilledBoardIndex() const {
        __constant__ static constexpr uint16_t KilledBoardIndexMask = Bit4 << 8;

        return (_packedIndexes & KilledBoardIndexMask) >> 8;
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetKilledBoardIndexCPU() const {
        static constexpr uint16_t KilledBoardIndexMask = Bit4 << 8;

        return (_packedIndexes & KilledBoardIndexMask) >> 8;
    }

    FAST_DCALL_ALWAYS void SetCastlingType(const uint16_t castlingType) { _packedIndexes |= castlingType << 12; }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetCastlingType() const {
        __constant__ static constexpr uint16_t CastlingTypeMask = Bit3 << 12;

        return (_packedIndexes & CastlingTypeMask) >> 12;
    }


    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetCastlingTypeCPU() const {
        static constexpr uint16_t CastlingTypeMask = Bit3 << 12;

        return (_packedIndexes & CastlingTypeMask) >> 12;
    }

    FAST_DCALL_ALWAYS void SetCheckType() {
        __constant__ static constexpr uint16_t CheckTypeBit = 1LLU << 15;

        _packedIndexes |= CheckTypeBit;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsChecking() const {
        __constant__ static constexpr uint16_t CheckTypeBit = 1LLU << 15;

        return (_packedIndexes & CheckTypeBit) > 0;
    }

    FAST_DCALL_ALWAYS void
    SetKilledFigureField(const uint16_t killedFigureField) { _packedMisc |= killedFigureField; }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetKilledFigureField() const {
        __constant__ static constexpr uint16_t KilledFigureFieldMask = Bit6;
        return _packedMisc & KilledFigureFieldMask;
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetKilledFigureFieldCPU() const {
        static constexpr uint16_t KilledFigureFieldMask = Bit6;
        return _packedMisc & KilledFigureFieldMask;
    }

    FAST_DCALL_ALWAYS void SetElPassantField(const uint16_t elPassantField) { _packedMisc |= elPassantField << 6; }

    [[nodiscard]] FAST_DCALL_ALWAYS uint16_t GetElPassantField() const {
        __constant__ static constexpr uint16_t ElPassantFieldMask = Bit6 << 6;

        return (_packedMisc & ElPassantFieldMask) >> 6;
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint16_t GetElPassantFieldCPU() const {
        static constexpr uint16_t ElPassantFieldMask = Bit6 << 6;

        return (_packedMisc & ElPassantFieldMask) >> 6;
    }

    FAST_DCALL_ALWAYS void SetCastlingRights(const uint32_t arr) {
        const uint16_t rights = arr & static_cast<uint32_t>(0xF);

        _packedMisc |= rights << 12;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS uint32_t GetCastlingRights() const {
        __constant__ static constexpr uint32_t CastlingMask = Bit4 << 12;

        const uint32_t rights = (_packedMisc & CastlingMask) >> 12;

        return rights;
    }

    [[nodiscard]] FAST_CALL_ALWAYS uint32_t GetCastlingRightsCPU() const {
        static constexpr uint32_t CastlingMask = Bit4 << 12;

        const uint32_t rights = (_packedMisc & CastlingMask) >> 12;

        return rights;
    }

    [[nodiscard]] uint16_t GetPackedIndexes() const { return _packedIndexes; }

    [[nodiscard]] uint16_t GetPackedMisc() const { return _packedMisc; }

    // ------------------------------
    // Private class methods
    // ------------------------------

    // ------------------------------
    // Class fields
    // ------------------------------

private:
    cuda_PackedMove _packedMove{};
    uint16_t _packedIndexes{};
    uint16_t _packedMisc{};
};

#endif // CUDA_MOVE_CUH
