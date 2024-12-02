//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef CUDA_MOVE_CUH
#define CUDA_MOVE_CUH

#include <cuda_runtime.h>
#include <string>

#include "cuda_Board.cuh"
#include "cuda_PackedBoard.cuh"

/*      Class encodes chess move and heuristic evaluation
 *  together inside some __uint64_t value. Encoding corresponds to value below:
 *  (bit indexed from lsb to msb manner)
 *  - bits 52-56 - encodes castling rights (4 bits) - "CastlingRights" - OBLIG IMPORTANT: should always be filled, in
 * case no changes was done simply copy previous values
 */

/*      Class encodes chess move and heuristic evaluation
 *  together inside some __uint64_t value. Encoding corresponds to value below:
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

__device__ __constant__ static constexpr __uint16_t PromoFlag = 0b1000;
__device__ __constant__ static constexpr __uint16_t CaptureFlag = 0b100;
__device__ __constant__ static constexpr __uint16_t CastlingFlag = 0b10;
__device__ __constant__ static constexpr __uint16_t QueenFlag = 0;
__device__ __constant__ static constexpr __uint16_t RookFlag = 0b1;
__device__ __constant__ static constexpr __uint16_t BishopFlag = 0b10;
__device__ __constant__ static constexpr __uint16_t KnightFlag = 0b11;
__device__ __constant__ static constexpr __uint16_t PromoSpecBits = 0b11;

__device__ __constant__ static constexpr __uint16_t MoveTypeBits = 0xF << 12;
__device__ __constant__ static constexpr __uint16_t Bit6 = 0b111111;
__device__ __constant__ static constexpr __uint16_t Bit4 = 0b1111;
__device__ __constant__ static constexpr __uint16_t Bit3 = 0b111;
__device__ __constant__ static constexpr __uint16_t PromoBit = PromoFlag << 12;
__device__ __constant__ static constexpr __uint16_t CaptureBit = CaptureFlag << 12;
__device__ __constant__ static constexpr __uint16_t CastlingBits = CastlingFlag << 12;

class cuda_Move;

inline std::pair<char, char> ConvertToCharPos(const int boardPosMsb) {
    const __uint32_t boardPos = ConvertToReversedPos(boardPosMsb);
    return {static_cast<char>('a' + (boardPos % 8)), static_cast<char>('1' + (boardPos / 8))};
}

struct cuda_PackedMove final {
    // ------------------------------
    // Class creation
    // ------------------------------

    FAST_CALL explicit cuda_PackedMove(const __uint16_t packedMove) : _packedMove(packedMove) {}

    cuda_PackedMove() = default;

    ~cuda_PackedMove() = default;

    cuda_PackedMove(const cuda_PackedMove &other) = default;

    cuda_PackedMove &operator=(const cuda_PackedMove &other) = default;

    cuda_PackedMove(cuda_PackedMove &&other) = default;

    cuda_PackedMove &operator=(cuda_PackedMove &&other) = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] std::string GetLongAlgebraicNotation() const {
        static constexpr char PromoFigs[] = {'q', 'r', 'b', 'n'};
        std::string rv;

        auto [c1, c2] = ConvertToCharPos((int) GetStartFieldCPU());
        rv += c1;
        rv += c2;
        auto [c3, c4] = ConvertToCharPos((int) GetTargetFieldCPU());
        rv += c3;
        rv += c4;

        if (IsPromo())
            rv += PromoFigs[GetMoveType() & PromoSpecBits];

        return rv;
    }

    FAST_CALL friend bool operator==(const cuda_PackedMove a, const cuda_PackedMove b) {
        return a._packedMove == b._packedMove;
    }

    FAST_CALL friend bool operator!=(const cuda_PackedMove a, const cuda_PackedMove b) { return !(a == b); }

    FAST_DCALL_ALWAYS void SetStartField(const __uint16_t startField) { _packedMove |= startField; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetStartField() const {
        __constant__ static constexpr __uint16_t StartFieldMask = Bit6;
        return StartFieldMask & _packedMove;
    }

    [[nodiscard]] FAST_CALL_ALWAYS __uint16_t GetStartFieldCPU() const {
        static constexpr __uint16_t StartFieldMask = Bit6;
        return StartFieldMask & _packedMove;
    }

    FAST_DCALL_ALWAYS void SetTargetField(const __uint16_t targetField) { _packedMove |= targetField << 6; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetTargetField() const {
        __constant__ static constexpr __uint16_t TargetFieldMask = Bit6 << 6;
        return (_packedMove & TargetFieldMask) >> 6;
    }

    [[nodiscard]] FAST_CALL_ALWAYS __uint16_t GetTargetFieldCPU() const {
        static constexpr __uint16_t TargetFieldMask = Bit6 << 6;
        return (_packedMove & TargetFieldMask) >> 6;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsEmpty() const { return _packedMove == 0; }

    [[nodiscard]] FAST_DCALL_ALWAYS  bool IsQuiet() const { return (_packedMove & MoveTypeBits) == 0; }

    [[nodiscard]] FAST_DCALL_ALWAYS  bool IsCapture() const { return (_packedMove & CaptureBit) != 0; }

    [[nodiscard]] FAST_CALL_ALWAYS  bool IsPromo() const { return (_packedMove & PromoBit) != 0; }

    [[nodiscard]] FAST_DCALL_ALWAYS  bool IsCastling() const { return (_packedMove & MoveTypeBits) == CastlingBits; }

    FAST_DCALL_ALWAYS void SetMoveType(const __uint16_t MoveType) { _packedMove |= MoveType << 12; }

    [[nodiscard]] FAST_CALL_ALWAYS __uint16_t GetMoveType() const { return (_packedMove & MoveTypeBits) >> 12; }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsValidMove() const { return !IsEmpty(); }

    // debugging tool
    [[nodiscard]] FAST_DCALL_ALWAYS bool IsOkayMove() const {
        return !IsEmpty() && GetTargetField() != GetStartField();
    }

    [[nodiscard]] __uint16_t Dump() const { return _packedMove; }

    // ------------------------------
    // Class fields
    // ------------------------------

private:

    friend cuda_Move;

    __uint16_t _packedMove{};
};

/* Class used to preserve some crucial board state between moves */
struct VolatileBoardData {
    using _fetcher_t = cuda_PackedBoard<PACKED_BOARD_DEFAULT_SIZE>::BoardFetcher;

    VolatileBoardData() = delete;

    FAST_CALL explicit VolatileBoardData(const cuda_Board &bd)
            : Castlings(bd.Castlings), OldElPassant(bd.ElPassantField) {
    }

    FAST_CALL explicit VolatileBoardData(__uint32_t c, __uint64_t ep)
            : Castlings(c), OldElPassant(ep) {
    }

    FAST_CALL explicit VolatileBoardData(const _fetcher_t &fetcher)
            : Castlings(fetcher.Castlings()), OldElPassant(fetcher.ElPassantField()) {
    }

    const __uint32_t Castlings;
    const __uint64_t OldElPassant;
};

__device__ __constant__ static constexpr __uint32_t move_CastlingIdxArr[5]{
        SENTINEL_BOARD_INDEX, W_ROOK_INDEX, W_ROOK_INDEX, B_ROOK_INDEX, B_ROOK_INDEX
};

__device__ __constant__ static constexpr __uint64_t move_CastlingNewKingPos[5]{
        1LLU, CASTLING_NEW_ROOK_MAPS[0], CASTLING_NEW_ROOK_MAPS[1],
        CASTLING_NEW_ROOK_MAPS[2], CASTLING_NEW_ROOK_MAPS[3]
};

class cuda_Move final {
public:
// ------------------------------
// Class creation
// ------------------------------

    // This construction does not initialize crucial fields what must be done
    FAST_DCALL_ALWAYS explicit cuda_Move(const cuda_PackedMove mv) : _packedMove(mv) {}

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

    FAST_DCALL_ALWAYS void SetMoveType(const __uint16_t MoveType) { _packedMove.SetMoveType(MoveType); }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetMoveType() const { return _packedMove.GetMoveType(); }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsQuietMove() const {
        return !_packedMove.IsCapture() && !_packedMove.IsPromo();
    }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsValidMove() const { return _packedMove.IsValidMove(); }

    // debugging tool
    [[nodiscard]] FAST_DCALL_ALWAYS bool IsOkayMove() const { return _packedMove.IsOkayMove(); }

    template<__uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
    FAST_DCALL_ALWAYS static void MakeMove(const cuda_Move mv, cuda_PackedBoard<NUM_BOARDS>::BoardFetcher fetcher) {
        // removing the old piece from the board
        fetcher.SetBitBoard(fetcher.BitBoard(mv.GetStartBoardIndex()) ^ (cuda_MaxMsbPossible >> mv.GetStartField()),
                            mv.GetStartBoardIndex());

        // placing the figure on new field
        fetcher.SetBitBoard(fetcher.BitBoard(mv.GetTargetBoardIndex()) | (cuda_MaxMsbPossible >> mv.GetTargetField()),
                            mv.GetTargetBoardIndex());

        // removing the killed figure in case no figure is killed index should be indicating to the sentinel
        fetcher.SetBitBoard(
                fetcher.BitBoard(mv.GetKilledBoardIndex()) ^ (cuda_MaxMsbPossible >> mv.GetKilledFigureField()),
                mv.GetKilledBoardIndex());

        // applying new castling rights
        fetcher.Castlings() = mv.GetCastlingRights();

        // applying new el passant field
        fetcher.SetElPassantField(cuda_MaxMsbPossible >> mv.GetElPassantField());

        // applying additional castling operation
        const __uint32_t boardIndex = move_CastlingIdxArr[mv.GetCastlingType()];
        const __uint64_t newKingPos = move_CastlingNewKingPos[mv.GetCastlingType()];
        fetcher.SetBitBoard(fetcher.BitBoard(boardIndex) | newKingPos, boardIndex);

        fetcher.ChangePlayingColor();
    }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsAttackingMove() const { return _packedMove.IsCapture(); }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsEmpty() const { return _packedMove.IsEmpty(); }

    template<__uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
    FAST_DCALL_ALWAYS static void UnmakeMove(const cuda_Move mv, cuda_PackedBoard<NUM_BOARDS>::BoardFetcher fetcher, const VolatileBoardData &data) {
        fetcher.ChangePlayingColor();

        // placing the piece on old board
        fetcher.SetBitBoard(fetcher.BitBoard(mv.GetStartBoardIndex()) | (cuda_MaxMsbPossible >> mv.GetStartField()),
                            mv.GetStartBoardIndex());

        // removing the figure from the new field
        fetcher.SetBitBoard(fetcher.BitBoard(mv.GetTargetBoardIndex()) ^ (cuda_MaxMsbPossible >> mv.GetTargetField()),
                            mv.GetTargetBoardIndex());

        // placing the killed figure in good place
        fetcher.SetBitBoard(
                fetcher.BitBoard(mv.GetKilledBoardIndex()) | (cuda_MaxMsbPossible >> mv.GetKilledFigureField()),
                mv.GetKilledBoardIndex());

        // recovering old castlings
        fetcher.Castlings() = data.Castlings;

        // recovering old el passant field
        fetcher.SetElPassantField(data.OldElPassant);

        // reverting castling operation
        const __uint32_t boardIndex = move_CastlingIdxArr[mv.GetCastlingType()];
        const __uint64_t newKingPos = move_CastlingNewKingPos[mv.GetCastlingType()];

        fetcher.SetBitBoard(fetcher.BitBoard(boardIndex) ^ newKingPos, boardIndex);
    }

    FAST_DCALL_ALWAYS void SetStartField(const __uint16_t startField) { _packedMove.SetStartField(startField); }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetStartField() const { return _packedMove.GetStartField(); }

    FAST_DCALL_ALWAYS void SetTargetField(const __uint16_t targetField) { _packedMove.SetTargetField(targetField); }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetTargetField() const { return _packedMove.GetTargetField(); }

    FAST_DCALL_ALWAYS void SetStartBoardIndex(const __uint16_t startBoard) { _packedIndexes |= startBoard; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetStartBoardIndex() const {
        __constant__ static constexpr __uint16_t StartBoardMask = Bit4;

        return _packedIndexes & StartBoardMask;
    }

    FAST_DCALL_ALWAYS void SetTargetBoardIndex(const __uint16_t targetBoardIndex) {
        _packedIndexes |= targetBoardIndex << 4;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetTargetBoardIndex() const {
        __constant__ static constexpr __uint16_t TargetBoardIndexMask = Bit4 << 4;

        return (_packedIndexes & TargetBoardIndexMask) >> 4;
    }

    FAST_DCALL_ALWAYS void SetKilledBoardIndex(const __uint16_t killedBoardIndex) {
        _packedIndexes |= killedBoardIndex << 8;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetKilledBoardIndex() const {
        __constant__ static constexpr __uint16_t KilledBoardIndexMask = Bit4 << 8;

        return (_packedIndexes & KilledBoardIndexMask) >> 8;
    }

    FAST_DCALL_ALWAYS void SetCastlingType(const __uint16_t castlingType) { _packedIndexes |= castlingType << 12; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetCastlingType() const {
        __constant__ static constexpr __uint16_t CastlingTypeMask = Bit3 << 12;

        return (_packedIndexes & CastlingTypeMask) >> 12;
    }

    FAST_DCALL_ALWAYS void SetCheckType() {
        __constant__ static constexpr __uint16_t CheckTypeBit = 1LLU << 15;

        _packedIndexes |= CheckTypeBit;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS bool IsChecking() const {
        __constant__ static constexpr __uint16_t CheckTypeBit = 1LLU << 15;

        return (_packedIndexes & CheckTypeBit) > 0;
    }

    FAST_DCALL_ALWAYS void
    SetKilledFigureField(const __uint16_t killedFigureField) { _packedMisc |= killedFigureField; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetKilledFigureField() const {
        __constant__ static constexpr __uint16_t KilledFigureFieldMask = Bit6;
        return _packedMisc & KilledFigureFieldMask;
    }

    FAST_DCALL_ALWAYS void SetElPassantField(const __uint16_t elPassantField) { _packedMisc |= elPassantField << 6; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint16_t GetElPassantField() const {
        __constant__ static constexpr __uint16_t ElPassantFieldMask = Bit6 << 6;

        return (_packedMisc & ElPassantFieldMask) >> 6;
    }

    FAST_DCALL_ALWAYS void SetCastlingRights(const __uint32_t arr) {
        const __uint16_t rights = arr & static_cast<__uint32_t>(0xF);

        _packedMisc |= rights << 12;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint32_t GetCastlingRights() const {
        __constant__ static constexpr __uint32_t CastlingMask = Bit4 << 12;

        const __uint32_t rights = (_packedMisc & CastlingMask) >> 12;

        return rights;
    }

    [[nodiscard]] __uint16_t GetPackedIndexes() const { return _packedIndexes; }

    [[nodiscard]] __uint16_t GetPackedMisc() const { return _packedMisc; }

// ------------------------------
// Private class methods
// ------------------------------

// ------------------------------
// Class fields
// ------------------------------

private:

    cuda_PackedMove _packedMove{};
    __uint16_t _packedIndexes{};
    __uint16_t _packedMisc{};
};

#endif // CUDA_MOVE_CUH
