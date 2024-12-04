//
// Created by Jlisowskyy on 30/11/24.
//

#ifndef SRC_CUDA_PACKEDBOARD_H
#define SRC_CUDA_PACKEDBOARD_H

#include "Helpers.cuh"
#include "cuda_Board.cuh"
#include "cuda_Array.cuh"

#include <vector>
#include <cassert>

template<__uint32_t NUM_BOARDS>
struct alignas(128) cuda_PackedBoard {
    // ------------------------------
    // Inner types
    // ------------------------------

    struct BoardFetcher {
        BoardFetcher() = default;

        HYBRID explicit BoardFetcher(__uint32_t idx, cuda_PackedBoard *__restrict__ packedBoard) : _idx(idx),
                                                                                      _packedBoard(packedBoard) {}

        // ------------------------------
        // Getters
        // ------------------------------

        [[nodiscard]] FAST_CALL_ALWAYS constexpr __uint32_t &MovingColor() {
            return _packedBoard->MovingColor[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr const __uint32_t &MovingColor() const {
            return _packedBoard->MovingColor[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr __uint32_t &Castlings() {
            return _packedBoard->Castlings[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr const __uint32_t &Castlings() const {
            return _packedBoard->Castlings[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr __int32_t &MaterialEval() {
            return _packedBoard->MaterialEval[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr const __int32_t &MaterialEval() const {
            return _packedBoard->MaterialEval[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr __uint64_t ElPassantField() const {
            __uint64_t lo = _packedBoard->ElPassantField[_idx];
            __uint64_t hi = _packedBoard->ElPassantField[NUM_BOARDS + _idx];

            return lo | (hi << 32);
        }

        static constexpr __uint64_t MASK_32_BIT = 0xFFFFFFFF;
        FAST_CALL_ALWAYS void constexpr SetElPassantField(__uint64_t field) {
            __uint64_t lo = field & MASK_32_BIT;
            __uint64_t hi = (field >> 32) & MASK_32_BIT;

            _packedBoard->ElPassantField[_idx] = lo;
            _packedBoard->ElPassantField[NUM_BOARDS + _idx] = hi;
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr __uint64_t BitBoard(__uint32_t idx) const {
            __uint64_t lo = _packedBoard->Boards[NUM_BOARDS * idx + _idx];
            __uint64_t hi = _packedBoard->Boards[(NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT) + NUM_BOARDS * idx + _idx];

            return lo | (hi << 32);
        }

        FAST_CALL_ALWAYS constexpr void SetBitBoard(__uint64_t field, __uint32_t idx) {
            __uint64_t lo = field & MASK_32_BIT;
            __uint64_t hi = (field >> 32) & MASK_32_BIT;

            _packedBoard->Boards[NUM_BOARDS * idx + _idx] = lo;
            _packedBoard->Boards[(NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT) + NUM_BOARDS * idx + _idx] = hi;
        }

        // ------------------------------
        // Interactions
        // ------------------------------

        FAST_CALL_ALWAYS constexpr void ChangePlayingColor() {
            MovingColor() ^= 1;
        }

        [[nodiscard]] FAST_DCALL_ALWAYS constexpr __uint32_t GetKingMsbPos(const __uint32_t col) const {
            return ExtractMsbPos(BitBoard(col * BIT_BOARDS_PER_COLOR + KING_INDEX));
        }

        [[nodiscard]] FAST_DCALL_ALWAYS constexpr __uint64_t GetFigBoard(__uint32_t col, __uint32_t figDesc) const {
            return BitBoard(col * BIT_BOARDS_PER_COLOR + figDesc);
        }

        FAST_DCALL_ALWAYS constexpr void SetCastlingRight(__uint32_t castlingIndex, bool value) {
            SetBitBoardBit(Castlings(), castlingIndex, value);
        }

        [[nodiscard]] FAST_CALL_ALWAYS constexpr bool GetCastlingRight(__uint32_t castlingIndex) const {
            return Castlings() & (cuda_MinMsbPossible << castlingIndex);
        }

        // ------------------------------
        // Class fields
        // ------------------------------

    protected:
        __uint32_t _idx;
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

        for (__uint32_t idx = 0; idx < boards.size(); ++idx) {
            BoardFetcher fetcher(idx, this);

            fetcher.MovingColor() = boards[idx].MovingColor;
            fetcher.Castlings() = boards[idx].Castlings;
            fetcher.SetElPassantField(boards[idx].ElPassantField);

            for (__uint32_t boardIdx = 0; boardIdx < BIT_BOARDS_COUNT; ++boardIdx) {
                fetcher.SetBitBoard(boards[idx].BitBoards[boardIdx], boardIdx);
            }

            fetcher.MaterialEval() = boards[idx].EvaluateMaterial();
        }
    }

    // ------------------------------
    // class interaction
    // ------------------------------

    [[nodiscard]] FAST_CALL_ALWAYS constexpr BoardFetcher operator[](__uint32_t boardIdx) {
        return BoardFetcher(boardIdx, this);
    }

    [[nodiscard]] FAST_CALL_ALWAYS constexpr const BoardFetcher &operator[](__uint32_t boardIdx) const {
        return BoardFetcher(boardIdx, this);
    }

    // --------------------------------
    // Main processing components
    // --------------------------------

    alignas(32) cuda_Array<__uint32_t, NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT * 2> Boards;
    alignas(32) cuda_Array<__uint32_t, NUM_BOARDS * 2> ElPassantField;
    alignas(32) cuda_Array<__uint32_t, NUM_BOARDS> Castlings;
    alignas(32) cuda_Array<__uint32_t, NUM_BOARDS> MovingColor;
    alignas(32) cuda_Array<__int32_t, NUM_BOARDS> MaterialEval;
};

using DefaultPackedBoardT = cuda_PackedBoard<PACKED_BOARD_DEFAULT_SIZE>;
using SmallPackedBoardT = cuda_PackedBoard<1>;

using BYTE = __uint8_t;

#endif //SRC_CUDA_PACKEDBOARD_H
