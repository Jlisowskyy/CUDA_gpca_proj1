//
// Author: Jakub Lisowski 11.2024
//

#include "cli/cli.hpp"
#include "cpu_core/cpu_core.hpp"

int main() {
    cpu_core core{};
    core.init();

    cli cli{&core};
    cli.run();

    return 0;
}
