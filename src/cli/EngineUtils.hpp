//
// Created by Jlisowskyy on 12/27/23.
//

#ifndef ENGINETYPEDEFS_H
#define ENGINETYPEDEFS_H

#include <chrono>
#include <climits>
#include <cstdint>
#include <numeric>
#include <unordered_map>

#include "../data_structs/cpu_Board.hpp"

/*
 * Header gather various simple functions with different purposes,
 * that are not directly connected with any class inside the project.
 *
 * */


/* Should not be used when "singleBitBitMap" has more than one bit positive */
constexpr uint64_t RemovePiece(uint64_t &bitMap, uint64_t singleBitBitMap) { return bitMap ^= singleBitBitMap; }

constexpr uint64_t RemovePiece(uint64_t &bitMap, int msbPos) { return bitMap ^= (MaxMsbPossible >> msbPos); }

constexpr uint64_t ConvertMsbPosToBitMap(int msbPos) { return (MaxMsbPossible >> msbPos); }

/* functions returns 'moves' that are present on 'allowedMoves' BitMap */
constexpr uint64_t FilterMoves(uint64_t moves, uint64_t allowedMoves) { return moves & allowedMoves; }

// ------------------------------
// Those functions below allows to perform aligned alloc operations without any problems on windows+linux
// Because default windows implementation of C++ library does not support std::aligned_alloc function

void *AlignedAlloc(size_t alignment, size_t size);
void AlignedFree(void *ptr);

// ------------------------------

/* Function simply prints given uint64_t as 8x8 block of 'x' chars when there is positive bit */
/* Prints it according to internal BitBoard indexing order */
void DisplayMask(uint64_t mask);

/* Function simply pretty prints the given board */
void DisplayBoard(const cpu_Board &bd);

/* Function simply returns sting containing information about the current datetime, mostly used in logging*/
std::string GetCurrentTimeStr();

/* Functions extract the name of file given inside the 'path' variable */
const char *GetFileName(const char *path);

// returns 0 if invalid characters were passed
uint64_t ExtractPosFromStr(int x, int y);

// in case of invalid encoding returns 0 in the corresponding value, if string is to short returns (0, 0).
// return scheme: [ oldMap, newMap ]
std::pair<uint64_t, uint64_t> ExtractPositionsFromEncoding(const std::string &encoding);

// Move to strings converting functions
std::pair<char, char> ConvertToCharPos(int boardPosMsb);

/* Function converts the given integer to string encoding the field accordingly to typical chess notation [A-H][1-8]*/
std::string ConvertToStrPos(int boardPosMsb);

/* Does same thing as function above, but additionally extracts the MsbPos from the BitBoard */
std::string ConvertToStrPos(uint64_t boardMap);

/* Table maps BitBoard index to character representing corresponding chess figure */
extern const char IndexToFigCharMap[cpu_Board::BitBoardsCount];

/* Does the same thing as above but in reverse */
/* This object must be inline due to problem with extern initialisation on startup  */
/* When FenTranslator referred empty not initialised map on its static field init */
inline static const std::unordered_map<char, size_t> FigCharToIndexMap{
    {'P',   wPawnsIndex},
    {'N', wKnightsIndex},
    {'B', wBishopsIndex},
    {'R',   wRooksIndex},
    {'Q',  wQueensIndex},
    {'K',    wKingIndex},
    {'p',   bPawnsIndex},
    {'n', bKnightsIndex},
    {'b', bBishopsIndex},
    {'r',   bRooksIndex},
    {'q',  bQueensIndex},
    {'k',    bKingIndex}
};

#endif // ENGINETYPEDEFS_H
