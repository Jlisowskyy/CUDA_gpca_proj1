//
// Created by Jlisowskyy on 15/11/24.
//

#ifndef SRC_BOARDDEFS_HPP
#define SRC_BOARDDEFS_HPP

#include <cstdint>

/*
 * Given enum defines values and order of both colors. All indexing schemes used across the projects follows given
 * order. It is important to keep it consistent across the project. It is also very useful when performing some indexing
 * and color switching operations.
 * */

enum Color : int {
    WHITE,
    BLACK,
};

/*
 * Given enum defines indexing order of all BitBoards inside the board.
 * It is important to keep this order consistent across the project.
 * Again very useful when performing some indexing and color switching operations.
 * */

enum ColorlessDescriptors : uint32_t {
    pawnsIndex,
    knightsIndex,
    bishopsIndex,
    rooksIndex,
    queensIndex,
    kingIndex,
};

/*
 * Given enum defines indexing order of all (color, piece) BitBoards inside the board.
 * It is important to keep it consistent across the project.
 * Used rather less frequently than previous ones but still defines order of all bitboards.
 * */

enum Descriptors : uint32_t {
    wPawnsIndex,
    wKnightsIndex,
    wBishopsIndex,
    wRooksIndex,
    wQueensIndex,
    wKingIndex,
    bPawnsIndex,
    bKnightsIndex,
    bBishopsIndex,
    bRooksIndex,
    bQueensIndex,
    bKingIndex,
};

/*
 * Defines the order of castling indexes for given color.
 * */

enum CastlingIndexes : uint32_t {
    KingCastlingIndex,
    QueenCastlingIndex,
};

/*
 * Defines indexes of all castling possibilities.
 * */

enum CastlingPossibilities : uint32_t {
    WhiteKingSide,
    WhiteQueenSide,
    BlackKingSide,
    BlackQueenSide,
};

#endif //SRC_BOARDDEFS_HPP
