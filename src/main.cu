//
// Author: Jakub Lisowski 11.2024
//

#include "cli/cli.hpp"
#include "cpu_core/cpu_core.cuh"

int main() {
    cpu_core core{};
    core.init();

    cli cli{&core};
    cli.run();

    return 0;
}
