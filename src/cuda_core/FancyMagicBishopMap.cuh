//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_FANCYMAGICBISHOPMAP_CUH
#define SRC_FANCYMAGICBISHOPMAP_CUH

#include <thrust/pair.h>

#include "BishopMapGenerator.cuh"
#include "BaseMoveHashMap.cuh"
#include "cuda_Array.cuh"
#include "cuda_Board.cuh"

__device__ static constexpr cuda_Array<__uint64_t, BitBoardFields> BISHOP_PARAMS_MAGICS{
        2459020380749179396LLU,
        18228596997040662761LLU,
        577023771530035456LLU,
        140742052938256LLU,
        4241559519926323204LLU,
        40533590659958800LLU,
        13573396270746166127LLU,
        15433831379842884746LLU,
        5458137257786957413LLU,
        9011666189571585LLU,
        72128546918695297LLU,
        3378867985317888LLU,
        6397499356811370666LLU,
        565157583650897LLU,
        2452495878734481824LLU,
        2305992547226353793LLU,
        5896716211751964931LLU,
        4652306653314547856LLU,
        162130170776389698LLU,
        11613292506489506816LLU,
        12494788516674558978LLU,
        288804355648307712LLU,
        5001228016187988525LLU,
        167298237218849LLU,
        7397172614331155718LLU,
        15882937967404034260LLU,
        40541527747936518LLU,
        4069501445375656064LLU,
        10083843241836806529LLU,
        6266797948639643237LLU,
        586034268766736388LLU,
        2308394151070795840LLU,
        10054851337067495605LLU,
        90353743543543936LLU,
        1189092139733139972LLU,
        12483134911496110085LLU,
        1223299049642410168LLU,
        4888956467406968000LLU,
        145246620002288640LLU,
        12456504737662980LLU,
        576601500060224000LLU,
        4619649999728459497LLU,
        5975714137870106383LLU,
        2932970949829467478LLU,
        2902574806744450484LLU,
        1688859317807109378LLU,
        9368613263417540672LLU,
        18307229971900796838LLU,
        9223443788588130432LLU,
        9228491680955777025LLU,
        10905209255181491571LLU,
        4415763513441LLU,
        5042597873833943191LLU,
        144418812266088580LLU,
        576495945332868224LLU,
        2314885401732973320LLU,
        1738429605722719888LLU,
        2395073496242704814LLU,
        2306125723593080833LLU,
        117392932654286848LLU,
        6726376817723420599LLU,
        1731642857114050572LLU,
        2306977739592179777LLU,
        18381131039969901408LLU,
};

__device__ static constexpr cuda_Array<__uint64_t, BitBoardFields> BISHOP_OFFSET_PARAMS{
        6,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        7,
        7,
        7,
        7,
        5,
        5,
        5,
        5,
        7,
        9,
        9,
        7,
        5,
        5,
        5,
        5,
        7,
        9,
        9,
        7,
        5,
        5,
        5,
        5,
        7,
        7,
        7,
        7,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        5,
        5,
        5,
        5,
        5,
        5,
        6
};


class FancyMagicBishopMap {
    using _underlyingMapT = BaseMoveHashMap<MaxPossibleNeighborsWithOverlap>;

public:
    __device__ constexpr FancyMagicBishopMap() {
        for (int i = 0; i < static_cast<int>(BitBoardFields); ++i) {
            const int boardIndex = ConvertToReversedPos(i);
            const __uint64_t magic = BISHOP_PARAMS_MAGICS[i];
            const __uint64_t shift = BISHOP_OFFSET_PARAMS[i];

            _maps[i] = _underlyingMapT(BishopMapGenerator::InitMasks(boardIndex), magic, shift);

            MoveInitializer(
                    _maps[i],
                    [](const __uint64_t n, const int ind) constexpr {
                        return BishopMapGenerator::GenMoves(n, ind);
                    },
                    []([[maybe_unused]] const int, const BishopMapGenerator::MasksT &m) constexpr {
                        return BishopMapGenerator::GenPossibleNeighborsWithOverlap(m);
                    },
                    [](const __uint64_t b, const BishopMapGenerator::MasksT &m) constexpr {
                        return BishopMapGenerator::StripBlockingNeighbors(b, m);
                    },
                    boardIndex
            );
        }
    }

    [[nodiscard]] __device__ constexpr __uint64_t GetMoves(int msbInd, __uint64_t fullBoard) const {
        const __uint64_t neighbors = fullBoard & _maps[msbInd].getFullMask();
        return _maps[msbInd][neighbors];
    }

    // ------------------------------
    // class fields
    // ------------------------------

protected:
    cuda_Array<_underlyingMapT, BitBoardFields> _maps;
};

#endif //SRC_FANCYMAGICBISHOPMAP_CUH
