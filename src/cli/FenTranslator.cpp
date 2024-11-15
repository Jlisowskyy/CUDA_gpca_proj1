//
// Created by Jlisowskyy on 12/27/23.
//

#include "FenTranslator.hpp"

#include <iostream>
#include <format>

#include "ParseTools.hpp"
#include "../move_gen/table_based/tables/BlackPawnMap.hpp"
#include "../move_gen/table_based/tables/WhitePawnMap.hpp"

bool FenTranslator::Translate(const std::string &fenPos, cpu_Board &bd)
{
    cpu_Board workBoard{};

    try
    {
        size_t pos = 0;
        pos        = _skipBlanks(pos, fenPos);
        pos        = _processPositions(workBoard, pos, fenPos);
        pos        = _skipBlanks(pos, fenPos);
        pos        = _processMovingColor(workBoard, pos, fenPos);
        pos        = _skipBlanks(pos, fenPos);
        pos        = _processCastlings(workBoard, pos, fenPos);
        pos        = _skipBlanks(pos, fenPos);
        pos        = _processElPassant(workBoard, pos, fenPos);
        pos        = _skipBlanks(pos, fenPos);
    }
    catch (const std::exception &exc)
    {
        std::cout << (exc.what()) << '\n';
        std::cout << ("[ INFO ] Loading default layout...\n");
        return false;
    }

    bd = workBoard;
    return true;
}

std::string FenTranslator::Translate(const cpu_Board &board)
{
    std::string fenPos{};
    _extractFiguresEncoding(board, fenPos);
    fenPos += ' ';
    fenPos += board.MovingColor == WHITE ? 'w' : 'b';
    fenPos += ' ';
    fenPos += _extractCastling(board);
    fenPos += ' ';

    // inner representation points to position made with long pawn move
    const auto FenCompatibleElPassantPosition = board.MovingColor == WHITE
                                                    ? WhitePawnMap::GetElPassantMoveField(board.ElPassantField)
                                                    : BlackPawnMap::GetElPassantMoveField(board.ElPassantField);

    fenPos +=
            board.ElPassantField == cpu_Board::InvalidElPassantBitBoard ? "-" : ConvertToStrPos(FenCompatibleElPassantPosition);

//    fenPos += ' ';
//    fenPos += std::to_string(board.HalfMoves);
//    fenPos += ' ';
//    fenPos += std::to_string((board.Age + 1) / 2);

    fenPos += ' ';
    fenPos += '0';
    fenPos += ' ';
    fenPos += '0';

    return fenPos;
}

std::string FenTranslator::_extractCastling(const cpu_Board &bd)
{
    std::string str{};
    for (size_t i = 0; i < cpu_Board::CastlingCount; ++i)
        if (bd.GetCastlingRight(i))
            str += CastlingNames[i];

    if (str.empty())
        str = "-";
    return str;
}

void FenTranslator::_extractFiguresEncoding(const cpu_Board &bd, std::string &fenPos)
{
    int inSeries{};

    for (signed_size_t y = 7; y >= 0; --y)
    {
        for (signed_size_t x = 0; x < 8; ++x)
        {
            const int bInd = static_cast<int>(y * 8 + x);

            // reading figure from the board
            const auto [res, fig, col] = _extractSingleEncoding(bd, bInd);
            if (res == empty)
            {
                ++inSeries;
                continue;
            }

            // eventually adding empty figure FEN offset
            if (inSeries)
                fenPos += static_cast<char>('0' + inSeries);
            inSeries = 0;

            fenPos += col == WHITE ? static_cast<char>(std::toupper(fig)) : fig;
        }

        // eventually adding empty figure FEN offset
        if (inSeries)
            fenPos += static_cast<char>('0' + inSeries);
        inSeries = 0;

        // skipping last slash
        if (y != 0)
            fenPos += '/';
    }
}

std::tuple<FenTranslator::FieldOccup, char, Color>
FenTranslator::_extractSingleEncoding(const cpu_Board &bd, const int bInd)
{
    const uint64_t map = 1LLU << bInd;

    for (size_t i = 0; i < cpu_Board::BitBoardsCount; ++i)
    {
        if ((map & bd.BitBoards[i]) != 0)
        {
            Color col = i >= cpu_Board::BitBoardsPerCol ? BLACK : WHITE;
            char fig;

            switch (i % cpu_Board::BitBoardsPerCol)
            {
            case pawnsIndex:
                fig = 'p';
                break;
            case knightsIndex:
                fig = 'n';
                break;
            case bishopsIndex:
                fig = 'b';
                break;
            case rooksIndex:
                fig = 'r';
                break;
            case queensIndex:
                fig = 'q';
                break;
            case kingIndex:
                fig = 'k';
                break;
            }

            return {occupied, fig, col};
        }
    }

    return {empty, {}, {}};
}

size_t FenTranslator::_processElPassant(cpu_Board &bd, const size_t pos, const std::string &fenPos)
{
    if (pos >= fenPos.length())
        throw std::runtime_error("[ ERROR ] Fen position has invalid castling specified!\n");

    if (fenPos[pos] == '-')
    {
        // ElPassant substring consists of '-' character and some other unnecessary one.
        if (pos + 1 < fenPos.length() && !std::isblank(fenPos[pos + 1]))
            throw std::runtime_error("[ ERROR ] Fen position has invalid castling specified!\n");

        return pos + 1;
    }

    if (pos + 1 >= fenPos.length() || (pos + 2 < fenPos.length() && !std::isblank(fenPos[pos + 2])))
        throw std::runtime_error("[ ERROR ] Invalid field description detected on ElPassant field!\n");

    const auto field        = fenPos.substr(pos, 2);
    const uint64_t boardPos = ExtractPosFromStr(field[0], field[1]);

    if (boardPos == 0)
        throw std::runtime_error("[ ERROR ] Invalid field description detected on ElPassant field!\n");

    bd.ElPassantField = boardPos;

    // inner representation points to position made with long pawn move
    bd.ElPassantField = bd.MovingColor == WHITE ? BlackPawnMap::GetElPassantMoveField(bd.ElPassantField)
                                                : WhitePawnMap::GetElPassantMoveField(bd.ElPassantField);

    return pos + 2;
}

size_t FenTranslator::_processCastlings(cpu_Board &bd, size_t pos, const std::string &fenPos)
{
    if (pos >= fenPos.length())
        throw std::runtime_error("[ ERROR ] Fen position does not contain information about castling!\n");

    // Castlings not possible!
    if (fenPos[pos] == '-')
    {
        // Castling substring consists of '-' character and some other unnecessary one.
        if (pos + 1 < fenPos.length() && !std::isblank(fenPos[pos + 1]))
            throw std::runtime_error("[ ERROR ] Fen position has invalid castling specified!\n");

        return pos + 1;
    }

    // Processing possible castlings
    int processedPos = 0;
    while (pos < fenPos.length() && !std::isblank(fenPos[pos]))
    {
        size_t ind;

        switch (fenPos[pos])
        {
        case 'K':
            ind = WhiteKingSide;
            break;
        case 'k':
            ind = BlackKingSide;
            break;
        case 'Q':
            ind = WhiteQueenSide;
            break;
        case 'q':
            ind = BlackQueenSide;
            break;
        default:
            throw std::runtime_error(std::format("[ ERROR ] Unrecognized castling specifier: {0}!\n", fenPos[pos]));
        }

        if (bd.GetCastlingRight(ind))
        {
            throw std::runtime_error("[ ERROR ] Repeated same castling at least two time!\n");
        }

        bd.SetCastlingRight(ind, true);
        ++processedPos;
        ++pos;
    }

    // Empty substring
    if (processedPos == 0)
        throw std::runtime_error("[ ERROR ] Castling possiblities not specified!\n");

    return pos;
}

size_t FenTranslator::_processMovingColor(cpu_Board &bd, const size_t pos, const std::string &fenPos)
{
    // Too short fenPos string or too long moving color specyfing substring.
    if (pos >= fenPos.length() || (pos + 1 < fenPos.length() && !std::isblank(fenPos[pos + 1])))
        throw std::runtime_error("[ ERROR ] Fen position has invalid moving color specified!\n");

    // Color detection.
    if (fenPos[pos] == 'w')
        bd.MovingColor = WHITE;
    else if (fenPos[pos] == 'b')
        bd.MovingColor = BLACK;
    else
        throw std::runtime_error(
            std::format("[ ERROR ] Fen position contains invalid character ({0}) on moving color field", fenPos[pos])
        );

    return pos + 1;
}

size_t FenTranslator::_processPositions(cpu_Board &bd, size_t pos, const std::string &fenPos)
{
    int processedFields   = 0;
    std::string posBuffer = "00";

    while (pos < fenPos.length() && !std::isblank(fenPos[pos]))
    {
        // Encoding position into typical chess notation.
        posBuffer[0] = static_cast<char>(processedFields % 8 + 'a');
        posBuffer[1] = static_cast<char>('8' - (processedFields / 8));

        // Checking possibilites of enocuntered characters.
        // '/' can be safely omitted due to final fields counting with processedFields variable.
        if (const char val = fenPos[pos]; val >= '1' && val <= '8')
            processedFields += val - '0';
        else if (val != '/')
        {
            _addFigure(posBuffer, val, bd);
            ++processedFields;
        }

        if (processedFields > static_cast<int>(cpu_Board::BitBoardFields))
            throw std::runtime_error(std::format("[ ERROR ] Too much fields are used inside passed fen position!\n"));

        ++pos;
    }

    if (processedFields < static_cast<int>(cpu_Board::BitBoardFields))
        throw std::runtime_error(std::format("[ ERROR ] Not enugh fields are used inside passed fen position!\n"));

    return pos;
}

void FenTranslator::_addFigure(const std::string &pos, char fig, cpu_Board &bd)
{
    const auto field = ExtractPosFromStr(pos[0], pos[1]);

    if (field == 0)
        throw std::runtime_error(
            std::format("[ ERROR ] Encountered invalid character ({})inside fen position description!\n", pos)
        );

    if (!FigCharToIndexMap.contains(fig))
        throw std::runtime_error(
            std::format("[ ERROR ] Encountered invalid character ({}) inside fen position description!\n", fig)
        );

    bd.BitBoards[FigCharToIndexMap.at(fig)] |= field;
}

size_t FenTranslator::_skipBlanks(size_t pos, const std::string &fenPos)
{
    while (pos < fenPos.length() && std::isblank(fenPos[pos]))
    {
        ++pos;
    }
    return pos;
}

size_t FenTranslator::_processMovesCounts(size_t pos, const std::string &fenPos, int &counter)
{
    static constexpr auto convert = [](const std::string &str) -> int
    {
        return std::stoi(str);
    };

    size_t rv = ParseTools::ExtractNextNumeric<int, convert>(fenPos, pos, counter);

    if (rv == ParseTools::InvalidNextWorldRead)
        throw std::runtime_error("cuda_Move counter has not been found!");

    return rv;
}

cpu_Board FenTranslator::GetTranslated(const std::string &fenPos)
{
    cpu_Board rv;
    if (!Translate(fenPos, rv))
        rv = GetDefault();
    return rv;
}
