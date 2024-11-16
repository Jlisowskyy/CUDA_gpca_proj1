//
// Author: Jakub Lisowski 11.2024
//

#include "cli/Cli.hpp"
#include "cpu_core/CpuCore.cuh"

//#include "cuda_core/FancyMagicBishopMap.cuh"
#include "cuda_core/FancyMagicRookMap.cuh"

int main() {
    CpuCore core{};
    core.init();

    Cli cli{&core};
    cli.run();

    return 0;
}
