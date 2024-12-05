//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CPU_CORE_H
#define SRC_CPU_CORE_H

struct cudaDeviceProp;
struct cuda_Board;

#include <tuple>
#include "../cuda_core/Move.cuh"

class CpuCore final {
    // ------------------------------
    // Class creation
    // ------------------------------
public:

    CpuCore();

    ~CpuCore();

    // ------------------------------
    // Class interaction
    // ------------------------------

    void setBoard(cuda_Board *board);

    void runCVC(__uint32_t moveTime);

    void runPVC(__uint32_t moveTime, __uint32_t playerColor);

    void runInfinite();

    void init();

    [[nodiscard]] int getDeviceThreads() const {
        return m_deviceThreads;
    }

    [[nodiscard]] const cudaDeviceProp &getDeviceProps() const {
        return *m_deviceProps;
    }

    // ------------------------------
    // Class private methods
    // ------------------------------
private:

    [[nodiscard]] std::tuple<int, int, cudaDeviceProp *> _pickGpu();

    void _dumpGPUInfo(int idx, const cudaDeviceProp &props);

    [[nodiscard]] static cuda_Move _readPlayerMove(const std::vector<cuda_Move>& correctMoves);

    [[nodiscard]] static bool _validateMove(const std::vector<cuda_Move>& validMoves, cuda_Move move);

    static void _runProcessingAnim(__uint32_t moveTime);

    static void _waitForInfiniteStop();

    // ------------------------------
    // Class fields
    // ------------------------------

    cuda_Board *m_board{};

    int m_deviceThreads{};

    cudaDeviceProp *m_deviceProps{};
};


#endif //SRC_CPU_CORE_H
