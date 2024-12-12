#ifndef SRC_CUDA_PACKEDBOARD_H
#define SRC_CUDA_PACKEDBOARD_H

#include "Helpers.cuh"
#include "cuda_Board.cuh"
#include "cuda_Array.cuh"

#include <vector>
#include <cassert>

template<uint32_t NUM_BOARDS>
struct alignas(128) cuda_PackedBoard {
    // ------------------------------
    // Inner types
    // ------------------------------

    struct BoardFetcher {
        BoardFetcher() = default;

        HYBRID explicit BoardFetcher(uint32_t idx, cuda_PackedBoard *__restrict__ packedBoard) : _idx(idx),
            _packedBoard(packedBoard) {
        }

        // ------------------------------
        // Getters
        // ------------------------------

        [[nodiscard]] FAST_CALL_ALWAYS constexpr uint32_t &HalfMoves() {
            return _packedBoard->HalfMoves[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr const uint32_t &HalfMoves() const {
            return _packedBoard->HalfMoves[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr uint32_t &MovingColor() {
            return _packedBoard->MovingColor[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr const uint32_t &MovingColor() const {
            return _packedBoard->MovingColor[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr uint32_t &Castlings() {
            return _packedBoard->Castlings[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr const uint32_t &Castlings() const {
            return _packedBoard->Castlings[_idx];
        }

        [[nodiscard]] FAST_DCALL_ALWAYS constexpr int32_t &MaterialEval() {
            return _packedBoard->MaterialEval[_idx];
        }

        [[nodiscard]] FAST_DCALL_ALWAYS constexpr const int32_t &MaterialEval() const {
            return _packedBoard->MaterialEval[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr uint64_t ElPassantField() const {
            uint64_t lo = _packedBoard->ElPassantField[_idx];
            uint64_t hi = _packedBoard->ElPassantField[NUM_BOARDS + _idx];

            return lo | (hi << 32);
        }

        static constexpr uint64_t MASK_32_BIT = 0xFFFFFFFF;
        FAST_CALL_ALWAYS void constexpr SetElPassantField(uint64_t field) {
            uint64_t lo = field & MASK_32_BIT;
            uint64_t hi = (field >> 32) & MASK_32_BIT;

            _packedBoard->ElPassantField[_idx] = lo;
            _packedBoard->ElPassantField[NUM_BOARDS + _idx] = hi;
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr uint64_t BitBoard(uint32_t idx) const {
            uint64_t lo = _packedBoard->Boards[NUM_BOARDS * idx + _idx];
            uint64_t hi = _packedBoard->Boards[(NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT) + NUM_BOARDS * idx + _idx];

            return lo | (hi << 32);
        }

        FAST_CALL_ALWAYS constexpr void SetBitBoard(uint64_t field, uint32_t idx) {
            uint64_t lo = field & MASK_32_BIT;
            uint64_t hi = (field >> 32) & MASK_32_BIT;

            _packedBoard->Boards[NUM_BOARDS * idx + _idx] = lo;
            _packedBoard->Boards[(NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT) + NUM_BOARDS * idx + _idx] = hi;
        }

        // ------------------------------
        // Interactions
        // ------------------------------

        FAST_CALL_ALWAYS constexpr void ChangePlayingColor() {
            MovingColor() ^= 1;
        }

        [[nodiscard]] FAST_DCALL_ALWAYS constexpr uint32_t GetKingMsbPos(const uint32_t col) const {
            return ExtractMsbPos(BitBoard(col * BIT_BOARDS_PER_COLOR + KING_INDEX));
        }

        [[nodiscard]] FAST_DCALL_ALWAYS constexpr uint64_t GetFigBoard(uint32_t col, uint32_t figDesc) const {
            return BitBoard(col * BIT_BOARDS_PER_COLOR + figDesc);
        }

        FAST_DCALL_ALWAYS constexpr void SetCastlingRight(uint32_t castlingIndex, bool value) {
            SetBitBoardBit(Castlings(), castlingIndex, value);
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr bool GetCastlingRight(uint32_t castlingIndex) const {
            return Castlings() & (cuda_MinMsbPossible << castlingIndex);
        }

        [[nodiscard]] int32_t FAST_DCALL_ALWAYS EvaluateMaterial() const {
            int32_t eval{};

            for (uint32_t bIdx = 0; bIdx < BIT_BOARDS_COUNT; ++bIdx) {
                eval += CountOnesInBoard(BitBoard(bIdx)) * FIG_VALUES[bIdx];
            }

            return eval;
        }

        [[nodiscard]] int32_t FAST_DCALL_ALWAYS EvaluateMaterial(const uint32_t color) const {
            const int32_t eval = EvaluateMaterial();
            return color == BLACK ? -eval : eval;
        }


        // ------------------------------
        // Class fields
        // ------------------------------

    protected:
        uint32_t _idx;
        cuda_PackedBoard *__restrict__ _packedBoard;
    };

    // ------------------------------
    // Class creation
    // ------------------------------

    cuda_PackedBoard() = default;

    ~cuda_PackedBoard() = default;

    cuda_PackedBoard(const cuda_PackedBoard &) = default;

    cuda_PackedBoard &operator=(const cuda_PackedBoard &) = default;

    explicit cuda_PackedBoard(const std::vector<cuda_Board> &boards) {
        assert(boards.size() <= NUM_BOARDS);

        for (uint32_t idx = 0; idx < boards.size(); ++idx) {
            saveBoard(idx, boards[idx]);
        }
    }

    // ------------------------------
    // class interaction
    // ------------------------------

    INLINE void saveBoard(const uint32_t idx, const cuda_Board &board) {
        assert(idx < NUM_BOARDS && "DETECTED OVERFLOW");
        BoardFetcher fetcher(idx, this);

        fetcher.MovingColor() = board.MovingColor;
        fetcher.Castlings() = board.Castlings;
        fetcher.SetElPassantField(board.ElPassantField);

        for (uint32_t boardIdx = 0; boardIdx < BIT_BOARDS_COUNT; ++boardIdx) {
            fetcher.SetBitBoard(board.BitBoards[boardIdx], boardIdx);
        }

        fetcher.MaterialEval() = board.EvaluateMaterial();
        fetcher.HalfMoves() = board.HalfMoves;
    }

    [[nodiscard]] FAST_CALL_ALWAYS constexpr BoardFetcher operator[](uint32_t boardIdx) {
        assert(boardIdx < NUM_BOARDS && "DETECTED OVERFLOW");

        return BoardFetcher(boardIdx, this);
    }

    [[nodiscard]] FAST_CALL_ALWAYS constexpr const BoardFetcher &operator[](uint32_t boardIdx) const {
        assert(boardIdx < NUM_BOARDS && "DETECTED OVERFLOW");

        return BoardFetcher(boardIdx, this);
    }

    // --------------------------------
    // Main processing components
    // --------------------------------

    alignas(32) cuda_Array<uint32_t, NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT * 2> Boards;
    alignas(32) cuda_Array<uint32_t, NUM_BOARDS * 2> ElPassantField;
    alignas(32) cuda_Array<uint32_t, NUM_BOARDS> Castlings;
    alignas(32) cuda_Array<uint32_t, NUM_BOARDS> MovingColor;
    alignas(32) cuda_Array<int32_t, NUM_BOARDS> MaterialEval;
    alignas(32) cuda_Array<uint32_t, NUM_BOARDS> HalfMoves;
};

using DefaultPackedBoardT = cuda_PackedBoard<PACKED_BOARD_DEFAULT_SIZE>;
using SmallPackedBoardT = cuda_PackedBoard<1>;

using BYTE = uint8_t;

#endif //SRC_CUDA_PACKEDBOARD_H
