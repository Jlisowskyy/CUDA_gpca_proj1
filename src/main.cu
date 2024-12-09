//
// Author: Jakub Lisowski 11.2024
//

#include "cpu_core/Cli.cuh"
#include "cpu_core/CpuCore.cuh"
#include "cpu_core/Utils.cuh"

#include "tests/CudaTests.cuh"
#include "ported/CpuUtils.h"

#include <string>
#include <iostream>

#include "cpu_core/GlobalState.cuh"

int main(const int argc, const char *argv[]) {
    CpuCore::parseFlags(argc, argv);

    InitializeConsole();

    CpuCore core{};
    core.init();

    if (g_GlobalState.RunTestsOnStart) {
        if (g_CudaTestsMap.contains(g_GlobalState.TestName)) {
            const auto &[_, __, testFunc] = g_CudaTestsMap.at(g_GlobalState.TestName);
            testFunc(core.getDeviceThreads(), core.getDeviceProps());
        } else {
            std::cerr << "Test not found" << std::endl;
            return 1;
        }

        return 0;
    }

    Cli cli{&core};
    cli.run();

    return 0;
}
