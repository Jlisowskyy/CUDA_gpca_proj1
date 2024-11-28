//
// Created by Jlisowskyy on 15/11/24.
//

#include "Helpers.cuh"
#include "cuda_Board.cuh"

#include <iostream>
#include <format>
#include <thrust/pair.h>

void AssertSuccess(cudaError_t error, const char *file, int line) {
    TraceError(error, file, line);

    if (error != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
}

bool TraceError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        std::cerr << std::format("CUDA Error at {}:{} - {}\n", file, line, cudaGetErrorString(error)) << std::endl;
    }

    return error != cudaSuccess;
}

HYBRID static thrust::pair<char, char> ConvertToCharPos(const __uint32_t boardPosMsb) {
    const __uint32_t boardPos = ConvertToReversedPos(boardPosMsb);
    return {static_cast<char>('a' + (boardPos % 8)), static_cast<char>('1' + (boardPos / 8))};
}

__device__ const char IndexToFigCharMap[BIT_BOARDS_COUNT]{
        'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k',
};

HYBRID void DisplayBoard(const cuda_Board *board) {
    static constexpr __uint32_t LastRowIndex = 7;
    static constexpr __uint32_t CharsForFig = 3;
    static constexpr __uint32_t FigsPerRow = 8;
    static constexpr __uint32_t FigsPerCol = 8;

    for (__uint32_t y = 0; y < FigsPerCol; ++y) {
        for (__uint32_t x = 0; x < FigsPerRow; ++x) {
            const uint64_t field = 1LLU << (y*8 + x);
            printf(" ");

            bool found = false;
            for (__uint32_t desc = 0; desc < BIT_BOARDS_COUNT; ++desc) {
                if ((board->BitBoards[desc] & field) != 0) {
                    printf("%c", IndexToFigCharMap[desc]);
                    found = true;
                    break;
                }
            }

            if (!found) {
                printf(" ");
            }
            printf(" ");

            if (x != LastRowIndex) {
                printf("|");
            } else {
                printf("   %c", static_cast<char>('8' - y));
            }
        }

        printf("\n");
        if (y != LastRowIndex) {
            for (__uint32_t idx = 0; idx < LastRowIndex + CharsForFig * FigsPerRow; ++idx) {
                printf("-");
            }
            printf("\n");
        }
    }

    for (__uint32_t idx = 0; idx < LastRowIndex + CharsForFig * FigsPerRow; ++idx) {
        printf("-");
    }
    printf("\n");

    for (__uint32_t x = 0; x < FigsPerRow; ++x) {
        printf(" %c  ", static_cast<char>('A' + x));
    }
    printf("\n");

    printf("Moving color: %s\n", board->MovingColor == WHITE ? "white" : "black");
    printf("Possible castlings:\n");
    static constexpr const char *castlingNames[] = {
            "White King Side", "White Queen Side", "Black King Side", "Black Queen Side"
    };

    for (__uint32_t i = 0; i < CASTLING_COUNT; ++i) {
        printf("%s: %d\n", castlingNames[i], board->GetCastlingRight(i));
    }


    const __uint32_t msbPos = ExtractMsbPosNeutral(board->ElPassantField);
    auto [c1, c2] = ConvertToCharPos(msbPos);
    char str[] = {c1, c2, '\0'};

    printf("El passant field: %s\n",
           (board->ElPassantField == INVALID_EL_PASSANT_BIT_BOARD ? "-" : str));
}
