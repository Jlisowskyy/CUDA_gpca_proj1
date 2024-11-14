//
// Created by Jlisowskyy on 12/28/23.
//

#include "EngineUtils.hpp"
#include "CompilationConstants.hpp"

#include <cstring>
#include <format>
#include <iostream>

const char IndexToFigCharMap[Board::BitBoardsCount]{
        'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k',
};

void DisplayMask(const uint64_t mask) {
    for (int y = 56; y >= 0; y -= 8) {
        for (int x = 0; x < 8; ++x) {
            const uint64_t pos = 1LLU << (y + x);

            std::cout << ' ' << ((pos & mask) != 0 ? 'x' : ' ') << ' ' << (x != 7 ? '|' : '\n');
        }

        if (y != 0) {
            std::cout << std::string(7 + 3 * 8, '-') << std::endl;
        }
    }
}

void DisplayBoard(const Board &bd) {
    static constexpr size_t LastRowIndex = 7;
    static constexpr size_t CharsForFig = 3;
    static constexpr size_t FigsPerRow = 8;
    static constexpr size_t FigsPerCol = 8;

    for (size_t y = 0; y < FigsPerCol; ++y) {
        for (size_t x = 0; x < FigsPerRow; ++x) {
            const uint64_t field = ExtractPosFromStr(static_cast<char>(x + 'a'), static_cast<char>('8' - y));

            std::cout << ' ';

            bool found = false;
            for (size_t desc = 0; desc < Board::BitBoardsCount; ++desc) {
                if ((bd.BitBoards[desc] & field) != 0) {
                    std::cout << IndexToFigCharMap[desc];
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << ' ';
            }
            std::cout << ' ';

            if (x != LastRowIndex) {
                std::cout << '|';
            } else {
                std::cout << std::format("   {}", static_cast<char>('8' - y));
            }
        }

        std::cout << std::endl;
        if (y != LastRowIndex) {
            std::cout << std::string(LastRowIndex + CharsForFig * FigsPerRow, '-') << std::endl;
        }
    }

    std::cout << std::string(LastRowIndex + CharsForFig * FigsPerRow, '-') << std::endl;
    for (size_t x = 0; x < FigsPerRow; ++x) {
        std::cout << ' ' << static_cast<char>('A' + x) << ' ' << ' ';
    }
    std::cout << std::endl;

    std::cout << "Moving color: " << (bd.MovingColor == WHITE ? "white" : "black") << std::endl;
    std::cout << "Possible castlings:\n";
    static constexpr const char *castlingNames[] = {
            "White King Side", "White Queen Side", "Black King Side", "Black Queen Side"
    };
    for (size_t i = 0; i < Board::CastlingCount; ++i) {
        std::cout << castlingNames[i] << ": " << bd.Castlings[i] << std::endl;
    }

    std::cout << "El passant field: "
              << (bd.ElPassantField == Board::InvalidElPassantBitBoard
                  ? "-"
                  : ConvertToStrPos(bd.ElPassantField))
              << std::endl;
}

uint64_t ExtractPosFromStr(int x, const int y) {
    x = tolower(x);

    if (x > 'h' || x < 'a')
        return 0;
    if (y < '1' || y > '8')
        return 0;

    return 1LLU << ((y - '1') * 8 + (x - 'a'));
}

std::pair<uint64_t, uint64_t> ExtractPositionsFromEncoding(const std::string &encoding) {
    if (encoding.length() < 4)
        return {0, 0};

    return {
            ExtractPosFromStr(std::toupper(encoding[0]), std::toupper(encoding[1])),
            ExtractPosFromStr(std::toupper(encoding[2]), std::toupper(encoding[3]))
    };
}

std::pair<char, char> ConvertToCharPos(const int boardPosMsb) {
    const int boardPos = ConvertToReversedPos(boardPosMsb);
    return {static_cast<char>('a' + (boardPos % 8)), static_cast<char>('1' + (boardPos / 8))};
}

std::string ConvertToStrPos(const int boardPosMsb) {
    static constexpr size_t PosStrSize = 2;
    std::string rv{PosStrSize};
    auto [c1, c2] = ConvertToCharPos(boardPosMsb);
    rv[0] = c1;
    rv[1] = c2;
    return rv;
}

std::string ConvertToStrPos(const uint64_t boardMap) { return ConvertToStrPos(ExtractMsbPos(boardMap)); }

void *AlignedAlloc(const size_t alignment, const size_t size) {
#ifdef __WIN32__
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void AlignedFree(void *ptr) {
#ifdef __WIN32__
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

std::string GetCurrentTimeStr() {
    static constexpr size_t BuffSize = 128;
    char buff[BuffSize]{};

    auto tm = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::strftime(buff, BuffSize, "%H:%M:%S", std::localtime(&tm));

    return std::string{buff};
}

const char *GetFileName(const char *path) {
    const char *lastSlash = std::strrchr(path, SLASH);
    return lastSlash == nullptr ? path : lastSlash + 1;
}
