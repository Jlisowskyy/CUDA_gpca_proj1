//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CPU_CORE_H
#define SRC_CPU_CORE_H

struct cudaDeviceProp;
struct cpu_Board;

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

    // ------------------------------
    // Class private methods
    // ------------------------------
private:

    std::pair<int, int> _pickGpu();

    void _dumpGPUInfo(int idx, const cudaDeviceProp &props);

    void _runSimpleMoveGen();

    // ------------------------------
    // Class fields
    // ------------------------------

    cpu_Board* m_board{};

    int m_deviceThreads{};
};


#endif //SRC_CPU_CORE_H
