#ifndef SRC_FANCYMAGICROOKMAP_CUH
#define SRC_FANCYMAGICROOKMAP_CUH

#include "cuda_Array.cuh"
#include "BaseMoveHashMap.cuh"
#include "RookMapGenerator.cuh"

__device__ static constexpr cuda_Array<uint64_t, BIT_BOARD_FIELDS> MAGICS_ROOK_PARAMS{
    1170940307609551394LLU,
    864693331908632740LLU,
    18577430269659234LLU,
    4621256752021644546LLU,
    4611721383624327186LLU,
    4631389405377988738LLU,
    2402707786700033LLU,
    9367534510373093457LLU,
    1191778255215034880LLU,
    283708634530816LLU,
    1190076220893593728LLU,
    288234784937214080LLU,
    27179931734188288LLU,
    158608849371392LLU,
    9148211623101056LLU,
    36073604282188288LLU,
    720576232445575169LLU,
    72620647137804339LLU,
    288276555707220096LLU,
    72066394427031564LLU,
    10150704251469832LLU,
    9007341006553152LLU,
    9011917007896576LLU,
    36046526648172545LLU,
    2314859571530694741LLU,
    13835075682431926929LLU,
    9800959238896353792LLU,
    2542073114792192LLU,
    704855681271808LLU,
    9011057804709890LLU,
    4527926326329920LLU,
    70369306214794LLU,
    4440998332545LLU,
    288511992862474244LLU,
    577024277784079440LLU,
    148623187905611776LLU,
    1229482732766167168LLU,
    306279963332448516LLU,
    9259611966951919616LLU,
    145275121385472LLU,
    1198467691593876LLU,
    9241390833545056354LLU,
    2342154380759531528LLU,
    1155314591798952960LLU,
    72198881550796800LLU,
    39549433821532224LLU,
    9227875911897071616LLU,
    4647715090393464896LLU,
    72198332616806528LLU,
    5774178050400911872LLU,
    140754668355968LLU,
    9223512808711281664LLU,
    4612249011331809312LLU,
    11854037274989707392LLU,
    108719710863888384LLU,
    130886003756958016LLU,
    4359484989186785536LLU,
    288232592490112169LLU,
    1225005641546989824LLU,
    9871908036182822144LLU,
    5044036051065126920LLU,
    612507141651046400LLU,
    18016666254314304LLU,
    612489824202555536LLU,
};

__device__ static constexpr cuda_Array<uint64_t, BIT_BOARD_FIELDS> OFFSETS_ROOK_PARAMS{
    12,
    11,
    11,
    11,
    11,
    11,
    11,
    12,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    12,
    11,
    11,
    11,
    11,
    11,
    11,
    12,
};

class FancyMagicRookMap final {
    using _underlyingMapT = BaseMoveHashMap<MaxRookPossibleNeighborsWithOverlap>;

public:
    constexpr FancyMagicRookMap() = default;

    static constexpr FancyMagicRookMap Create() {
        FancyMagicRookMap map{};

        for (int i = 0; i < static_cast<int>(BIT_BOARD_FIELDS); ++i) {
            const uint32_t boardIndex = ConvertToReversedPos(i);
            const uint64_t magic = MAGICS_ROOK_PARAMS[i];
            const uint64_t shift = OFFSETS_ROOK_PARAMS[i];

            map.m_maps[i] = _underlyingMapT(RookMapGenerator::InitMasks(boardIndex), magic, shift);

            MoveInitializer(
                map.m_maps[i],
                &RookMapGenerator::GenMoves,
                &RookMapGenerator::GenPossibleNeighborsWithOverlap,
                &RookMapGenerator::StripBlockingNeighbors,
                boardIndex
            );
        }

        return map;
    }

    constexpr FancyMagicRookMap(const FancyMagicRookMap &) = default;

    [[nodiscard]] FAST_DCALL_ALWAYS constexpr uint64_t GetMoves(uint32_t msbInd, uint64_t fullBoard) const {
        const uint64_t neighbors = fullBoard & m_maps[msbInd].getFullMask();
        return m_maps[msbInd][neighbors];
    }

    [[nodiscard]] cuda_Array<_underlyingMapT, BIT_BOARD_FIELDS> GetMaps() const {
        return m_maps;
    }

    // ------------------------------
    // class fields
    // ------------------------------

protected:
    alignas(128) cuda_Array<_underlyingMapT, BIT_BOARD_FIELDS> m_maps{};
};

#endif //SRC_FANCYMAGICROOKMAP_CUH
