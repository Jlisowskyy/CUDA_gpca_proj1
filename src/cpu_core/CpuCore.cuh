//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CPU_CORE_H
#define SRC_CPU_CORE_H

struct cudaDeviceProp;
struct cpu_Board;

#include <tuple>

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

    void setBoard(cpu_Board *board);

    void runCVC();

    void runPVC();

    void init();

    [[nodiscard]] int getDeviceThreads() const {
        return m_deviceThreads;
    }

    [[nodiscard]] const cudaDeviceProp& getDeviceProps() const {
        return *m_deviceProps;
    }

    // ------------------------------
    // Class private methods
    // ------------------------------
private:

    std::tuple<int, int, cudaDeviceProp *> _pickGpu();

    void _dumpGPUInfo(int idx, const cudaDeviceProp &props);

    void _runSimpleMoveGen();

    // ------------------------------
    // Class fields
    // ------------------------------

    cpu_Board* m_board{};

    int m_deviceThreads{};

    cudaDeviceProp *m_deviceProps{};
};


#endif //SRC_CPU_CORE_H
