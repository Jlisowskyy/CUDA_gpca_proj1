//
// Created by Jlisowskyy on 18/11/24.
//

#include "CudaTests.cuh"

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"
#include "../cuda_core/MoveGenerator.cuh"
#include "../cuda_core/cuda_Board.cuh"

#include "../ported/CpuTests.h"
#include "../ported/CpuUtils.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <string>
#include <string_view>

static constexpr std::string_view TestFEN[]{
//        "3r2k1/B7/4q3/1R4P1/1P3r2/8/2P2P1p/R4K2 w - - 0 49",
//        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
//        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
//        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
//        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
//        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
//        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
//        "3rr1k1/1pq2pp1/p2nb2p/2pp4/6PR/2PBPN2/PPQ2PP1/K2R4 b - - 0 20",
//        "7k/r2q1ppp/1p1p4/p1bPrPPb/P1PNPR1P/1PQ5/2B5/R5K1 w - - 23 16",
};

void __global__ GetGPUMovesKernel(const cuda_Board *board, cuda_Move *outMoves, int *outCount) {
    MoveGenerator gen{*board};

    const auto moves = gen.GetMovesFast();
    *outCount = static_cast<int>(moves.size);

    for (int i = 0; i < moves.size; ++i) {
        outMoves[i] = moves[i];
    }
}

void TestSinglePositionOutput(const std::string_view &fen) {
    std::cout << "Testing position: " << fen << std::endl;

    const auto externalBoard = cpu::TranslateFromFen(std::string(fen));
    const cuda_Board board(externalBoard);

    thrust::device_vector<cuda_Move> dMoves(256);
    thrust::device_vector<cuda_Board> dBoard{board};
    thrust::device_vector<int> dCount(1);

    GetGPUMovesKernel<<<1, 1>>>(thrust::raw_pointer_cast(dBoard.data()), thrust::raw_pointer_cast(dMoves.data()),
                                thrust::raw_pointer_cast(dCount.data()));
    const auto rc = cudaDeviceSynchronize();
    CUDA_TRACE_ERROR(rc);

    if (rc != cudaSuccess) {
        throw std::runtime_error("Failed to launch kernel");
    }

    cpu::external_board eBoard{};
    for (size_t i = 0; i < BitBoardsCount; ++i)
        eBoard[i] = board.BitBoards[i];
    eBoard[12] = board.ElPassantField;
    eBoard[13] = board.Castlings;
    eBoard[14] = board.MovingColor;

    const auto cMoves = cpu::GenerateMoves(eBoard);

    thrust::host_vector<cuda_Move> hMoves = dMoves;
    thrust::host_vector<int> hCount = dCount;

    if (cMoves.size() != hCount[0]) {
        std::cerr << "Moves count mismatch: " << cMoves.size() << " != " << hCount[0] << std::endl;
    }

    const size_t size = std::min(cMoves.size(), static_cast<size_t>(hCount[0]));

    __uint64_t errors = cMoves.size() - size;
    for (size_t i = 0; i < size; ++i) {
        const auto &cMove = cMoves[i];
        cuda_PackedMove ccMove{cMove[0]};
        const auto &hMove = hMoves[i];

        if (hMove.GetPackedMove() != ccMove) {
            std::cerr << "Packed move mismatch device:" << hMove.GetPackedMove().GetLongAlgebraicNotation() << " != "
                      << " host: " << ccMove.GetLongAlgebraicNotation() << std::endl;

            errors++;
        }

        if (hMove.GetPackedIndexes() != cMove[1]) {
            std::cerr << "Indexes mismatch device:" << hMove.GetPackedIndexes() << " != "
                      << " host: " << cMove[1] << std::endl;

            std::cerr << "Device: " << hMove.GetPackedMove().GetLongAlgebraicNotation() << std::endl;
            std::cerr << "Host: " << ccMove.GetLongAlgebraicNotation() << std::endl;

            errors++;
        }

        if (hMove.GetPackedMisc() != cMove[2]) {
            std::cerr << "Misc mismatch device:" << hMove.GetPackedMisc() << " != "
                      << " host: " << cMove[2] << std::endl;

            std::cerr << "Device: " << hMove.GetPackedMove().GetLongAlgebraicNotation() << std::endl;
            std::cerr << "Host: " << ccMove.GetLongAlgebraicNotation() << std::endl;

            errors++;
        }
    }

    if (errors != 0) {
        std::cerr << "Failed test for position: " << fen << std::endl;
        std::cerr << "Errors: " << errors << std::endl;

        std::cerr << "Device moves: " << std::endl;
        for (size_t i = 0; i < size; ++i) {
            std::cerr << hMoves[i].GetPackedMove().GetLongAlgebraicNotation() << std::endl;
        }

        std::cerr << "Host moves: " << std::endl;
        for (size_t i = 0; i < size; ++i) {
            std::cerr << cuda_PackedMove(cMoves[i][0]).GetLongAlgebraicNotation() << std::endl;
        }
    } else {
        std::cout << "Test passed for position: " << fen << std::endl;
    }
}

void MoveGenTest_(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    std::cout << "MoveGen Test" << std::endl;

    std::cout << "Testing positions with depth 0" << std::endl;
    for (const auto &fen: TestFEN) {
        TestSinglePositionOutput(fen);
    }
}

void MoveGenTest(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
        MoveGenTest_(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Failed test with Error: " << e.what() << std::endl;
    }
}
