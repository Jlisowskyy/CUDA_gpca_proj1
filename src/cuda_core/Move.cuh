//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef CUDA_MOVE_CUH
#define CUDA_MOVE_CUH

#include <cuda_runtime.h>

#include "cuda_Board.cuh"

/*      Class encodes chess move and heuristic evaluation
 *  together inside some uint64_t value. Encoding corresponds to value below:
 *  (bit indexed from lsb to msb manner)
 *  - bits 52-56 - encodes castling rights (4 bits) - "CastlingRights" - OBLIG IMPORTANT: should always be filled, in
 * case no changes was done simply copy previous values
 */

/*      Class encodes chess move and heuristic evaluation
 *  together inside some uint64_t value. Encoding corresponds to value below:
 *  - bits  0-15 - encodes simplistic heuristic evaluation of the move used inside inflight move sorting,
 *  - bits 16-31 - contains PackedMove instance
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

class alignas(16) Move;

struct alignas(16) PackedMove {
    // ------------------------------
    // Class creation
    // ------------------------------

    PackedMove() = default;

    ~PackedMove() = default;

    PackedMove(const PackedMove &other) = default;

    PackedMove &operator=(const PackedMove &other) = default;

    PackedMove(PackedMove &&other) = default;

    PackedMove &operator=(PackedMove &&other) = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    __device__ friend bool operator==(const PackedMove a, const PackedMove b) { return a._packedMove == b._packedMove; }

    __device__ friend bool operator!=(const PackedMove a, const PackedMove b) { return !(a == b); }

    __device__ void SetStartField(const uint16_t startField) { _packedMove |= startField; }

    __device__ uint16_t GetStartField() const {
        static constexpr uint16_t
        StartFieldMask = Bit6;
        return StartFieldMask & _packedMove;
    }

    __device__ void SetTargetField(const uint16_t targetField) { _packedMove |= targetField << 6; }

    __device__ uint16_t GetTargetField() const {
        static constexpr uint16_t
        TargetFieldMask = Bit6 << 6;
        return (_packedMove & TargetFieldMask) >> 6;
    }

    __device__ bool IsEmpty() const { return _packedMove == 0; }

    __device__  bool IsQuiet() const { return (_packedMove & MoveTypeBits) == 0; }

    __device__  bool IsCapture() const { return (_packedMove & CaptureBit) != 0; }

    __device__  bool IsPromo() const { return (_packedMove & PromoBit) != 0; }

    __device__  bool IsCastling() const { return (_packedMove & MoveTypeBits) == CastlingBits; }

    __device__ void SetMoveType(const uint16_t MoveType) { _packedMove |= MoveType << 12; }

    __device__  uint16_t GetMoveType() const { return (_packedMove & MoveTypeBits) >> 12; }

    __device__  bool IsValidMove() const { return !IsEmpty(); }

    // debugging tool
    __device__  bool IsOkeyMove() const { return !IsEmpty() && GetTargetField() != GetStartField(); }

    // ------------------------------
    // Class fields
    // ------------------------------

    static constexpr uint16_t
    PromoFlag = 0b1000;
    static constexpr uint16_t
    CaptureFlag = 0b100;
    static constexpr uint16_t
    CastlingFlag = 0b10;
    static constexpr uint16_t
    QueenFlag = 0;
    static constexpr uint16_t
    RookFlag = 0b1;
    static constexpr uint16_t
    BishopFlag = 0b10;
    static constexpr uint16_t
    KnightFlag = 0b11;
    static constexpr uint16_t
    PromoSpecBits = 0b11;

private:
    static constexpr uint16_t
    MoveTypeBits = 0xF << 12;
    static constexpr uint16_t
    Bit6 = 0b111111;

    static constexpr uint16_t
    PromoBit = PromoFlag << 12;
    static constexpr uint16_t
    CaptureBit = CaptureFlag << 12;

    static constexpr uint16_t
    CastlingBits = CastlingFlag << 12;

    friend Move;

    uint16_t _packedMove{};
};

/* Class used to preserve some crucial board state between moves */
struct VolatileBoardData {
    __device__ VolatileBoardData() = delete;

    __device__ explicit VolatileBoardData(const cuda_Board &bd)
            : Castlings(bd.Castlings), OldElPassant(bd.ElPassantField) {
    }

    const uint32_t Castlings;
    const uint64_t OldElPassant;
};

class alignas(16) Move {
public:
// ------------------------------
// Class creation
// ------------------------------

// This construction does not initialize crucial fields what must be done
    __device__ explicit Move(const PackedMove mv) : _packedMove(mv) {}

    Move() = default;

    ~Move() = default;

    Move(const Move &other) = default;

    Move &operator=(const Move &other) = default;

    Move(Move &&other) = default;

    Move &operator=(Move &&other) = default;

// ------------------------------
// Class interaction
// ------------------------------

    __device__ friend bool operator==(const Move a, const Move b) { return a._packedMove == b._packedMove; }

    __device__ friend bool operator!=(const Move a, const Move b) { return !(a == b); }

    __device__ PackedMove GetPackedMove() const { return _packedMove; }

    __device__ void SetMoveType(const uint16_t MoveType) { _packedMove.SetMoveType(MoveType); }

    __device__ uint16_t GetMoveType() const { return _packedMove.GetMoveType(); }

    __device__ bool IsQuietMove() const { return !_packedMove.IsCapture() && !_packedMove.IsPromo(); }

    __device__ bool IsValidMove() const { return _packedMove.IsValidMove(); }

    // debugging tool
    __device__ bool IsOkeyMove() const { return _packedMove.IsOkeyMove(); }

    __device__ static void MakeMove(const Move mv, cuda_Board &bd) {
        // removing the old piece from the board
        bd.BitBoards[mv.GetStartBoardIndex()] ^= MaxMsbPossible >> mv.GetStartField();

        // placing the figure on new field
        bd.BitBoards[mv.GetTargetBoardIndex()] |= MaxMsbPossible >> mv.GetTargetField();

        // removing the killed figure in case no figure is killed index should be indicating to the sentinel
        bd.BitBoards[mv.GetKilledBoardIndex()] ^= MaxMsbPossible >> mv.GetKilledFigureField();

        // applying new castling rights
        bd.Castlings = mv.GetCastlingRights();

        // applying new el passant field
        bd.ElPassantField = MaxMsbPossible >> mv.GetElPassantField();

        // applying additional castling operation
        const auto [boardIndex, field] = CastlingActions[mv.GetCastlingType()];
        bd.BitBoards[boardIndex] |= field;

        bd.ChangePlayingColor();
    }

    __device__ bool IsAttackingMove() const { return _packedMove.IsCapture(); }

    __device__ bool IsEmpty() const { return _packedMove.IsEmpty(); }

    __device__ static void UnmakeMove(const Move mv, cuda_Board &bd, const VolatileBoardData &data) {
        bd.ChangePlayingColor();

        // placing the piece on old board
        bd.BitBoards[mv.GetStartBoardIndex()] |= MaxMsbPossible >> mv.GetStartField();

        // removing the figure from the new field
        bd.BitBoards[mv.GetTargetBoardIndex()] ^= MaxMsbPossible >> mv.GetTargetField();

        // placing the killed figure in good place
        bd.BitBoards[mv.GetKilledBoardIndex()] |= MaxMsbPossible >> mv.GetKilledFigureField();

        // recovering old castlings
        bd.Castlings = data.Castlings;

        // recovering old el passant field
        bd.ElPassantField = data.OldElPassant;

        // reverting castling operation
        const auto [boardIndex, field] = CastlingActions[mv.GetCastlingType()];
        bd.BitBoards[boardIndex] ^= field;
    }

    __device__ void SetEval(const int32_t eval) { _eval = eval; }

    __device__ int32_t GetEval() const { return _eval; }

    __device__ void SetStartField(const uint16_t startField) { _packedMove.SetStartField(startField); }

    uint16_t GetStartField() const { return _packedMove.GetStartField(); }

    __device__ void SetTargetField(const uint16_t targetField) { _packedMove.SetTargetField(targetField); }

    __device__ uint16_t GetTargetField() const { return _packedMove.GetTargetField(); }

    __device__ void SetStartBoardIndex(const uint16_t startBoard) { _packedIndexes |= startBoard; }

    __device__ uint16_t GetStartBoardIndex() const {
        static constexpr uint16_t
        StartBoardMask = Bit4;
        return _packedIndexes & StartBoardMask;
    }

    __device__ void SetTargetBoardIndex(const uint16_t targetBoardIndex) { _packedIndexes |= targetBoardIndex << 4; }

    __device__ uint16_t GetTargetBoardIndex() const {
        static constexpr uint16_t
        TargetBoardIndexMask = Bit4 << 4;
        return (_packedIndexes & TargetBoardIndexMask) >> 4;
    }

    __device__ void SetKilledBoardIndex(const uint16_t killedBoardIndex) { _packedIndexes |= killedBoardIndex << 8; }

    __device__ uint16_t GetKilledBoardIndex() const {
        static constexpr uint16_t
        KilledBoardIndexMask = Bit4 << 8;
        return (_packedIndexes & KilledBoardIndexMask) >> 8;
    }

    __device__ void SetCastlingType(const uint16_t castlingType) { _packedIndexes |= castlingType << 12; }

    __device__ uint16_t GetCastlingType() const {
        static constexpr uint16_t
        CastlingTypeMask = Bit3 << 12;
        return (_packedIndexes & CastlingTypeMask) >> 12;
    }

    __device__ void SetCheckType() {
        static constexpr uint16_t
        CheckTypeBit = 1LLU << 15;
        _packedIndexes |= CheckTypeBit;
    }

    __device__ bool IsChecking() const {
        static constexpr uint16_t
        CheckTypeBit = 1LLU << 15;
        return (_packedIndexes & CheckTypeBit) > 0;
    }

    __device__ void SetKilledFigureField(const uint16_t killedFigureField) { _packedMisc |= killedFigureField; }

    __device__ uint16_t GetKilledFigureField() const {
        static constexpr uint16_t
        KilledFigureFieldMask = Bit6;
        return _packedMisc & KilledFigureFieldMask;
    }

    __device__ void SetElPassantField(const uint16_t elPassantField) { _packedMisc |= elPassantField << 6; }

    __device__ uint16_t GetElPassantField() const {
        static constexpr uint16_t
        ElPassantFieldMask = Bit6 << 6;
        return (_packedMisc & ElPassantFieldMask) >> 6;
    }

    __device__ void SetCasltingRights(const std::bitset<Board::CastlingCount + 1> arr) {
        const uint16_t rights = arr.to_ullong() & 0xFLLU;
        _packedMisc |= rights << 12;
    }

    __device__ std::bitset<Board::CastlingCount + 1> GetCastlingRights() const {
        static constexpr uint16_t
        CastlingMask = Bit4 << 12;
        const uint16_t rights = (_packedMisc & CastlingMask) >> 12;
        const std::bitset<Board::CastlingCount + 1> arr{rights};

        return arr;
    }

// ------------------------------
// Private class methods
// ------------------------------

// ------------------------------
// Class fields
// ------------------------------

private:
    static constexpr uint16_t
    Bit4 = 0b1111;
    static constexpr uint16_t
    Bit6 = 0b111111;
    static constexpr uint16_t
    Bit3 = 0b111;

    int32_t _eval{};
    PackedMove _packedMove{};
    uint16_t _packedIndexes{};
    uint16_t _packedMisc{};

public:
    static constexpr std::pair<size_t, uint64_t>
    CastlingActions[] = {
        {
            Board::SentinelBoardIndex, 1LLU
        },
        {
            wRooksIndex, Board::CastlingNewRookMaps[0]
        },
        {
            wRooksIndex, Board::CastlingNewRookMaps[1]
        },
        {
            bRooksIndex, Board::CastlingNewRookMaps[2]
        },
        {
            bRooksIndex, Board::CastlingNewRookMaps[3]
        },
    };
};

#endif // CUDA_MOVE_CUH
