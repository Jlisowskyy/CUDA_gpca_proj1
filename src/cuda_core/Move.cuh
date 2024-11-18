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

__device__ static constexpr uint16_t PromoFlag = 0b1000;
__device__ static constexpr uint16_t CaptureFlag = 0b100;
__device__ static constexpr uint16_t CastlingFlag = 0b10;
__device__ static constexpr uint16_t QueenFlag = 0;
__device__ static constexpr uint16_t RookFlag = 0b1;
__device__ static constexpr uint16_t BishopFlag = 0b10;
__device__ static constexpr uint16_t KnightFlag = 0b11;
__device__ static constexpr uint16_t PromoSpecBits = 0b11;

__device__ static constexpr uint16_t MoveTypeBits = 0xF << 12;
__device__ static constexpr uint16_t Bit6 = 0b111111;
__device__ static constexpr uint16_t Bit4 = 0b1111;
__device__ static constexpr uint16_t Bit3 = 0b111;
__device__ static constexpr uint16_t PromoBit = PromoFlag << 12;
__device__ static constexpr uint16_t CaptureBit = CaptureFlag << 12;
__device__ static constexpr uint16_t CastlingBits = CastlingFlag << 12;

class alignas(16) cuda_Move;

struct alignas(16) cuda_PackedMove final {
    // ------------------------------
    // Class creation
    // ------------------------------

    cuda_PackedMove() = default;

    ~cuda_PackedMove() = default;

    cuda_PackedMove(const cuda_PackedMove &other) = default;

    cuda_PackedMove &operator=(const cuda_PackedMove &other) = default;

    cuda_PackedMove(cuda_PackedMove &&other) = default;

    cuda_PackedMove &operator=(cuda_PackedMove &&other) = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    __device__ friend bool operator==(const cuda_PackedMove a, const cuda_PackedMove b) {
        return a._packedMove == b._packedMove;
    }

    __device__ friend bool operator!=(const cuda_PackedMove a, const cuda_PackedMove b) { return !(a == b); }

    __device__ void SetStartField(const uint16_t startField) { _packedMove |= startField; }

    __device__ uint16_t GetStartField() const {
        static constexpr uint16_t StartFieldMask = Bit6;
        return StartFieldMask & _packedMove;
    }

    __device__ void SetTargetField(const uint16_t targetField) { _packedMove |= targetField << 6; }

    __device__ uint16_t GetTargetField() const {
        static constexpr uint16_t TargetFieldMask = Bit6 << 6;
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

private:

    friend cuda_Move;

    uint16_t _packedMove{};
};

/* Class used to preserve some crucial board state between moves */
struct VolatileBoardData {
    VolatileBoardData() = delete;

    FAST_CALL explicit VolatileBoardData(const cuda_Board &bd)
            : Castlings(bd.Castlings), OldElPassant(bd.ElPassantField) {
    }

    const uint32_t Castlings;
    const uint64_t OldElPassant;
};

__device__ static constexpr size_t move_CastlingIdxArr[5]{
        SentinelBoardIndex, wRooksIndex, wRooksIndex, bRooksIndex, bRooksIndex
};

__device__ static constexpr uint64_t move_CastlingNewKingPos[5]{
        1LLU, CastlingNewRookMaps[0], CastlingNewRookMaps[1],
        CastlingNewRookMaps[2], CastlingNewRookMaps[3]
};

class alignas(16) cuda_Move final {
public:
// ------------------------------
// Class creation
// ------------------------------

    // This construction does not initialize crucial fields what must be done
    __device__ explicit cuda_Move(const cuda_PackedMove mv) : _packedMove(mv) {}

    cuda_Move() = default;

    ~cuda_Move() = default;

    cuda_Move(const cuda_Move &other) = default;

    cuda_Move &operator=(const cuda_Move &other) = default;

    cuda_Move(cuda_Move &&other) = default;

    cuda_Move &operator=(cuda_Move &&other) = default;

// ------------------------------
// Class interaction
// ------------------------------

    __device__ friend bool operator==(const cuda_Move a, const cuda_Move b) { return a._packedMove == b._packedMove; }

    __device__ friend bool operator!=(const cuda_Move a, const cuda_Move b) { return !(a == b); }

    __device__ cuda_PackedMove GetPackedMove() const { return _packedMove; }

    __device__ void SetMoveType(const uint16_t MoveType) { _packedMove.SetMoveType(MoveType); }

    __device__ uint16_t GetMoveType() const { return _packedMove.GetMoveType(); }

    __device__ bool IsQuietMove() const { return !_packedMove.IsCapture() && !_packedMove.IsPromo(); }

    __device__ bool IsValidMove() const { return _packedMove.IsValidMove(); }

    // debugging tool
    __device__ bool IsOkeyMove() const { return _packedMove.IsOkeyMove(); }

    __device__ static void MakeMove(const cuda_Move mv, cuda_Board &bd) {
        // removing the old piece from the board
        bd.BitBoards[mv.GetStartBoardIndex()] ^= cuda_MaxMsbPossible >> mv.GetStartField();

        // placing the figure on new field
        bd.BitBoards[mv.GetTargetBoardIndex()] |= cuda_MaxMsbPossible >> mv.GetTargetField();

        // removing the killed figure in case no figure is killed index should be indicating to the sentinel
        bd.BitBoards[mv.GetKilledBoardIndex()] ^= cuda_MaxMsbPossible >> mv.GetKilledFigureField();

        // applying new castling rights
        bd.Castlings = mv.GetCastlingRights();

        // applying new el passant field
        bd.ElPassantField = cuda_MaxMsbPossible >> mv.GetElPassantField();

        // applying additional castling operation
        const size_t boardIndex = move_CastlingIdxArr[mv.GetCastlingType()];
        const uint64_t newKingPos = move_CastlingNewKingPos[mv.GetCastlingType()];
        bd.BitBoards[boardIndex] |= newKingPos;

        bd.ChangePlayingColor();
    }

    __device__ bool IsAttackingMove() const { return _packedMove.IsCapture(); }

    __device__ bool IsEmpty() const { return _packedMove.IsEmpty(); }

    __device__ static void UnmakeMove(const cuda_Move mv, cuda_Board &bd, const VolatileBoardData &data) {
        bd.ChangePlayingColor();

        // placing the piece on old board
        bd.BitBoards[mv.GetStartBoardIndex()] |= cuda_MaxMsbPossible >> mv.GetStartField();

        // removing the figure from the new field
        bd.BitBoards[mv.GetTargetBoardIndex()] ^= cuda_MaxMsbPossible >> mv.GetTargetField();

        // placing the killed figure in good place
        bd.BitBoards[mv.GetKilledBoardIndex()] |= cuda_MaxMsbPossible >> mv.GetKilledFigureField();

        // recovering old castlings
        bd.Castlings = data.Castlings;

        // recovering old el passant field
        bd.ElPassantField = data.OldElPassant;

        // reverting castling operation
        const size_t boardIndex = move_CastlingIdxArr[mv.GetCastlingType()];
        const uint64_t newKingPos = move_CastlingNewKingPos[mv.GetCastlingType()];

        bd.BitBoards[boardIndex] ^= newKingPos;
    }

    __device__ void SetEval(const int32_t eval) { _eval = eval; }

    __device__ int32_t GetEval() const { return _eval; }

    __device__ void SetStartField(const uint16_t startField) { _packedMove.SetStartField(startField); }

    __device__ uint16_t GetStartField() const { return _packedMove.GetStartField(); }

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

    __device__ void SetCasltingRights(const uint32_t arr) {
        const uint16_t rights = arr & 0xFLLU;
        _packedMisc |= rights << 12;
    }

    __device__ uint32_t GetCastlingRights() const {
        static constexpr uint32_t
        CastlingMask = Bit4 << 12;
        const uint32_t rights = (_packedMisc & CastlingMask) >> 12;

        return rights;
    }

// ------------------------------
// Private class methods
// ------------------------------

// ------------------------------
// Class fields
// ------------------------------

private:

    int32_t _eval{};
    cuda_PackedMove _packedMove{};
    uint16_t _packedIndexes{};
    uint16_t _packedMisc{};
};

#endif // CUDA_MOVE_CUH
