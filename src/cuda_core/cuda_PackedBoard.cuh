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
struct cuda_PackedBoard {
    // ------------------------------
    // Inner types
    // ------------------------------

    struct BoardFetcher {
        explicit BoardFetcher(__uint32_t idx, cuda_PackedBoard &packedBoard) : _idx(idx), _packedBoard(packedBoard) {}

        // ------------------------------
        // Getters
        // ------------------------------

        [[nodiscard]] FAST_CALL_ALWAYS __uint32_t &MovingColor() {
            return _packedBoard.MovingColor[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS const __uint32_t &MovingColor() const {
            return _packedBoard.MovingColor[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS __uint32_t &Castlings() {
            return _packedBoard.Castlings[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS const __uint32_t &Castlings() const {
            return _packedBoard.Castlings[_idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS __uint64_t ElPassantField() const {
            __uint64_t lo = _packedBoard.ElPassantField[_idx];
            __uint64_t hi = _packedBoard.ElPassantField[NUM_BOARDS + _idx];

            return lo | (hi << 32);
        }

        void FAST_CALL_ALWAYS SetElPassantField(__uint64_t field) {
            static constexpr __uint64_t MASK = 0xFFFFFFFF;

            __uint64_t lo = field & MASK;
            __uint64_t hi = (field >> 32) & MASK;

            _packedBoard.ElPassantField[_idx] = lo;
            _packedBoard.ElPassantField[NUM_BOARDS + _idx] = hi;
        }

        [[nodiscard]] FAST_CALL_ALWAYS __uint64_t BitBoard(__uint32_t idx) const {
            __uint64_t lo = _packedBoard.Boards[NUM_BOARDS * idx + _idx];
            __uint64_t hi = _packedBoard.Boards[(NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT) + NUM_BOARDS * idx + _idx];

            return lo | (hi << 32);
        }

        void FAST_CALL_ALWAYS SetBitBoard(__uint64_t field, __uint32_t idx) {
            static constexpr __uint64_t MASK = 0xFFFFFFFF;

            __uint64_t lo = field & MASK;
            __uint64_t hi = (field >> 32) & MASK;

            _packedBoard.Boards[NUM_BOARDS * idx + _idx] = lo;
            _packedBoard.Boards[(NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT) + NUM_BOARDS * idx + _idx] = hi;
        }

        // ------------------------------
        // Interactions
        // ------------------------------

        FAST_CALL_ALWAYS void ChangePlayingColor() {
            MovingColor() ^= 1;
        }

        [[nodiscard]] FAST_DCALL_ALWAYS __uint32_t GetKingMsbPos(const __uint32_t col) const {
            return ExtractMsbPos(BitBoard(col * BIT_BOARDS_PER_COLOR + KING_INDEX));
        }

        [[nodiscard]] FAST_DCALL_ALWAYS __uint64_t GetFigBoard(__uint32_t col, __uint32_t figDesc) const {
            return BitBoard(col * BIT_BOARDS_PER_COLOR + figDesc);
        }

        FAST_DCALL_ALWAYS void SetCastlingRight(__uint32_t castlingIndex, bool value) {
            SetBitBoardBit(Castlings(), castlingIndex, value);
        }

        [[nodiscard]] FAST_CALL_ALWAYS bool GetCastlingRight(__uint32_t castlingIndex) const {
            return Castlings() & (cuda_MinMsbPossible << castlingIndex);
        }

        // ------------------------------
        // Class fields
        // ------------------------------

    protected:
        __uint32_t _idx;
        cuda_PackedBoard &_packedBoard;
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
            BoardFetcher fetcher(idx, *this);

            fetcher.MovingColor() = boards[idx].MovingColor;
            fetcher.Castlings() = boards[idx].Castlings;
            fetcher.SetElPassantField(boards[idx].ElPassantField);

            for (__uint32_t boardIdx = 0; boardIdx < BIT_BOARDS_COUNT; ++boardIdx) {
                fetcher.SetBitBoard(boards[idx].BitBoards[boardIdx], boardIdx);
            }
        }
    }

    // ------------------------------
    // class interaction
    // ------------------------------

    BoardFetcher operator[](__uint32_t boardIdx) {
        return BoardFetcher(boardIdx, *this);
    }

    const BoardFetcher &operator[](__uint32_t boardIdx) const {
        return BoardFetcher(boardIdx, *this);
    }

    // --------------------------------
    // Main processing components
    // --------------------------------

    cuda_Array<__uint32_t, NUM_BOARDS * BIT_BOARDS_GUARDED_COUNT * 2> Boards;
    cuda_Array<__uint32_t, NUM_BOARDS * 2> ElPassantField;
    cuda_Array<__uint32_t, NUM_BOARDS> Castlings;
    cuda_Array<__uint32_t, NUM_BOARDS> MovingColor;
};

#endif //SRC_CUDA_PACKEDBOARD_H
