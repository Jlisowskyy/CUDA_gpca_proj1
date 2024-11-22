//
// Author: Jakub Lisowski 11.2024
//

#include "cpu_core/Cli.cuh"
#include "cpu_core/CpuCore.cuh"

#include "tests/CudaTests.cuh"

#include <string>
#include <iostream>

#include "cuda_core/Move.cuh"
#include "cuda_core/Stack.cuh"

int main() {
    CpuCore core{};
    core.init();

#ifdef TESTING

    const std::string test = TESTING;

    if (CudaTestsMap.find(test) != CudaTestsMap.end()) {
        const auto &[_, __, testFunc] = CudaTestsMap.at(test);
        testFunc(core.getDeviceThreads(), core.getDeviceProps());
    } else {
        std::cerr << "Test not found" << std::endl;
    }

#else

    Cli cli{&core};
    cli.run();

#endif

    return 0;
}
