//
// Created by Jlisowskyy on 2/10/24.
//

#ifndef FANCYMAGICBISHOPMAP_H
#define FANCYMAGICBISHOPMAP_H

#include "BishopMapGenerator.hpp"
#include "BaseMoveHashMap.hpp"

class FancyMagicBishopMap {
    using _underlyingMapT = BaseMoveHashMap<BishopMapGenerator::MaxPossibleNeighborsWithOverlap>;

public:
    constexpr FancyMagicBishopMap();

    [[nodiscard]] constexpr uint64_t GetMoves(int msbInd, uint64_t fullBoard) const;


    // ------------------------------
    // class fields
    // ------------------------------

protected:
    std::array<_underlyingMapT, Board::BitBoardFields> _maps;

    static constexpr std::pair<uint64_t, uint64_t> PARAMS[Board::BitBoardFields]{
            std::make_pair(2459020380749179396LLU, 6),
            std::make_pair(18228596997040662761LLU, 5),
            std::make_pair(577023771530035456LLU, 5),
            std::make_pair(140742052938256LLU, 5),
            std::make_pair(4241559519926323204LLU, 5),
            std::make_pair(40533590659958800LLU, 5),
            std::make_pair(13573396270746166127LLU, 5),
            std::make_pair(15433831379842884746LLU, 6),
            std::make_pair(5458137257786957413LLU, 5),
            std::make_pair(9011666189571585LLU, 5),
            std::make_pair(72128546918695297LLU, 5),
            std::make_pair(3378867985317888LLU, 5),
            std::make_pair(6397499356811370666LLU, 5),
            std::make_pair(565157583650897LLU, 5),
            std::make_pair(2452495878734481824LLU, 5),
            std::make_pair(2305992547226353793LLU, 5),
            std::make_pair(5896716211751964931LLU, 5),
            std::make_pair(4652306653314547856LLU, 5),
            std::make_pair(162130170776389698LLU, 7),
            std::make_pair(11613292506489506816LLU, 7),
            std::make_pair(12494788516674558978LLU, 7),
            std::make_pair(288804355648307712LLU, 7),
            std::make_pair(5001228016187988525LLU, 5),
            std::make_pair(167298237218849LLU, 5),
            std::make_pair(7397172614331155718LLU, 5),
            std::make_pair(15882937967404034260LLU, 5),
            std::make_pair(40541527747936518LLU, 7),
            std::make_pair(4069501445375656064LLU, 9),
            std::make_pair(10083843241836806529LLU, 9),
            std::make_pair(6266797948639643237LLU, 7),
            std::make_pair(586034268766736388LLU, 5),
            std::make_pair(2308394151070795840LLU, 5),
            std::make_pair(10054851337067495605LLU, 5),
            std::make_pair(90353743543543936LLU, 5),
            std::make_pair(1189092139733139972LLU, 7),
            std::make_pair(12483134911496110085LLU, 9),
            std::make_pair(1223299049642410168LLU, 9),
            std::make_pair(4888956467406968000LLU, 7),
            std::make_pair(145246620002288640LLU, 5),
            std::make_pair(12456504737662980LLU, 5),
            std::make_pair(576601500060224000LLU, 5),
            std::make_pair(4619649999728459497LLU, 5),
            std::make_pair(5975714137870106383LLU, 7),
            std::make_pair(2932970949829467478LLU, 7),
            std::make_pair(2902574806744450484LLU, 7),
            std::make_pair(1688859317807109378LLU, 7),
            std::make_pair(9368613263417540672LLU, 5),
            std::make_pair(18307229971900796838LLU, 5),
            std::make_pair(9223443788588130432LLU, 5),
            std::make_pair(9228491680955777025LLU, 5),
            std::make_pair(10905209255181491571LLU, 5),
            std::make_pair(4415763513441LLU, 5),
            std::make_pair(5042597873833943191LLU, 5),
            std::make_pair(144418812266088580LLU, 5),
            std::make_pair(576495945332868224LLU, 5),
            std::make_pair(2314885401732973320LLU, 5),
            std::make_pair(1738429605722719888LLU, 6),
            std::make_pair(2395073496242704814LLU, 5),
            std::make_pair(2306125723593080833LLU, 5),
            std::make_pair(117392932654286848LLU, 5),
            std::make_pair(6726376817723420599LLU, 5),
            std::make_pair(1731642857114050572LLU, 5),
            std::make_pair(2306977739592179777LLU, 5),
            std::make_pair(18381131039969901408LLU, 6),
    };

};

constexpr FancyMagicBishopMap::FancyMagicBishopMap() {
    for (int i = 0; i < static_cast<int>(Board::BitBoardFields); ++i) {
        const int boardIndex = ConvertToReversedPos(i);
        const auto [magic, shift] = PARAMS[i];

        _maps[i] = _underlyingMapT(BishopMapGenerator::InitMasks(boardIndex), magic, shift);

        MoveInitializer(
                _maps[i],
                [](const uint64_t n, const int ind) constexpr {
                    return BishopMapGenerator::GenMoves(n, ind);
                },
                []([[maybe_unused]] const int, const BishopMapGenerator::MasksT &m) constexpr {
                    return BishopMapGenerator::GenPossibleNeighborsWithOverlap(m);
                },
                [](const uint64_t b, const BishopMapGenerator::MasksT &m) constexpr {
                    return BishopMapGenerator::StripBlockingNeighbors(b, m);
                },
                boardIndex
        );
    }
}

constexpr uint64_t FancyMagicBishopMap::GetMoves(const int msbInd, const uint64_t fullBoard) const {
    const uint64_t neighbors = fullBoard & _maps[msbInd].getFullMask();
    return _maps[msbInd][neighbors];
}

#endif // FANCYMAGICBISHOPMAP_H
