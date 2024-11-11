//
// Created by Jlisowskyy on 2/10/24.
//

#ifndef FANCYMAGICROOKMAP_H
#define FANCYMAGICROOKMAP_H

#include "RookMapGenerator.hpp"
#include "BaseMoveHashMap.hpp"
#include "../Board.hpp"

class FancyMagicRookMap {
    using _underlyingMapT = BaseMoveHashMap<RookMapGenerator::MaxRookPossibleNeighborsWithOverlap>;

public:
    constexpr FancyMagicRookMap();

    [[nodiscard]] constexpr uint64_t GetMoves(int msbInd, uint64_t fullBoard) const;

    // ------------------------------
    // class fields
    // ------------------------------

protected:

    std::array<_underlyingMapT, Board::BitBoardFields> m_maps{};

    static constexpr std::pair<uint64_t, uint64_t> PARAMS[Board::BitBoardFields]{
            std::make_pair(1170940307609551394LLU, 12),
            std::make_pair(864693331908632740LLU, 11),
            std::make_pair(18577430269659234LLU, 11),
            std::make_pair(4621256752021644546LLU, 11),
            std::make_pair(4611721383624327186LLU, 11),
            std::make_pair(4631389405377988738LLU, 11),
            std::make_pair(2402707786700033LLU, 11),
            std::make_pair(9367534510373093457LLU, 12),
            std::make_pair(1191778255215034880LLU, 11),
            std::make_pair(283708634530816LLU, 10),
            std::make_pair(1190076220893593728LLU, 10),
            std::make_pair(288234784937214080LLU, 10),
            std::make_pair(27179931734188288LLU, 10),
            std::make_pair(158608849371392LLU, 10),
            std::make_pair(9148211623101056LLU, 10),
            std::make_pair(36073604282188288LLU, 11),
            std::make_pair(720576232445575169LLU, 11),
            std::make_pair(72620647137804339LLU, 10),
            std::make_pair(288276555707220096LLU, 10),
            std::make_pair(72066394427031564LLU, 10),
            std::make_pair(10150704251469832LLU, 10),
            std::make_pair(9007341006553152LLU, 10),
            std::make_pair(9011917007896576LLU, 10),
            std::make_pair(36046526648172545LLU, 11),
            std::make_pair(2314859571530694741LLU, 11),
            std::make_pair(13835075682431926929LLU, 10),
            std::make_pair(9800959238896353792LLU, 10),
            std::make_pair(2542073114792192LLU, 10),
            std::make_pair(704855681271808LLU, 10),
            std::make_pair(9011057804709890LLU, 10),
            std::make_pair(4527926326329920LLU, 10),
            std::make_pair(70369306214794LLU, 11),
            std::make_pair(4440998332545LLU, 11),
            std::make_pair(288511992862474244LLU, 10),
            std::make_pair(577024277784079440LLU, 10),
            std::make_pair(148623187905611776LLU, 10),
            std::make_pair(1229482732766167168LLU, 10),
            std::make_pair(306279963332448516LLU, 10),
            std::make_pair(9259611966951919616LLU, 10),
            std::make_pair(145275121385472LLU, 11),
            std::make_pair(1198467691593876LLU, 11),
            std::make_pair(9241390833545056354LLU, 10),
            std::make_pair(2342154380759531528LLU, 10),
            std::make_pair(1155314591798952960LLU, 10),
            std::make_pair(72198881550796800LLU, 10),
            std::make_pair(39549433821532224LLU, 10),
            std::make_pair(9227875911897071616LLU, 10),
            std::make_pair(4647715090393464896LLU, 11),
            std::make_pair(72198332616806528LLU, 11),
            std::make_pair(5774178050400911872LLU, 10),
            std::make_pair(140754668355968LLU, 10),
            std::make_pair(9223512808711281664LLU, 10),
            std::make_pair(4612249011331809312LLU, 10),
            std::make_pair(11854037274989707392LLU, 10),
            std::make_pair(108719710863888384LLU, 10),
            std::make_pair(130886003756958016LLU, 11),
            std::make_pair(4359484989186785536LLU, 12),
            std::make_pair(288232592490112169LLU, 11),
            std::make_pair(1225005641546989824LLU, 11),
            std::make_pair(9871908036182822144LLU, 11),
            std::make_pair(5044036051065126920LLU, 11),
            std::make_pair(612507141651046400LLU, 11),
            std::make_pair(18016666254314304LLU, 11),
            std::make_pair(612489824202555536LLU, 12),
    };
};

constexpr FancyMagicRookMap::FancyMagicRookMap() {
    for (int i = 0; i < static_cast<int>(Board::BitBoardFields); ++i) {
        const int boardIndex = ConvertToReversedPos(i);
        const auto [magic, shift] = PARAMS[i];

        m_maps[i] = _underlyingMapT(RookMapGenerator::InitMasks(boardIndex), magic, shift);

        MoveInitializer(
                m_maps[i],
                [](const uint64_t n, const int ind) constexpr {
                    return RookMapGenerator::GenMoves(n, ind);
                },
                []([[maybe_unused]] const int, const RookMapGenerator::MasksT &m) constexpr {
                    return RookMapGenerator::GenPossibleNeighborsWithOverlap(m);
                },
                [](const uint64_t b, const RookMapGenerator::MasksT &m) constexpr {
                    return RookMapGenerator::StripBlockingNeighbors(b, m);
                },
                boardIndex
        );
    }
}

constexpr uint64_t FancyMagicRookMap::GetMoves(const int msbInd, const uint64_t fullBoard) const {
    const uint64_t neighbors = fullBoard & m_maps[msbInd].getFullMask();
    return m_maps[msbInd][neighbors];
}

#endif // FANCYMAGICROOKMAP_H
