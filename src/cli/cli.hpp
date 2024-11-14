//
// Created by Jlisowskyy on 14/11/24.
//

#ifndef SRC_CLI_H
#define SRC_CLI_H

/* Forward declaration */
class cpu_core;

class cli {
    // ------------------------------
    // Class creation
    // ------------------------------
public:

    explicit cli(cpu_core* core);
    ~cli();

    // ------------------------------
    // Class interaction
    // ------------------------------

    void run();

    // ------------------------------
    // Class fields
    // ------------------------------
private:

    cpu_core* m_core{};
};


#endif //SRC_CLI_H
