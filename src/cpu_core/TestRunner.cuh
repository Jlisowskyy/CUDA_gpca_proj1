//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_TESTRUNNER_CUH
#define SRC_TESTRUNNER_CUH

#include "../tests/CudaTests.cuh"

class CpuCore;

class TestRunner {
    enum class TestParseResult {
        SUCCESS,
        FAILURE,
        EXIT
    };

    // ------------------------------
    // Class creation
    // ------------------------------
public:

    explicit TestRunner(CpuCore *cpuCore);

    ~TestRunner();

    // ------------------------------
    // Class interaction
    // ------------------------------

    void runTests();

    // ------------------------------
    // Class private methods
    // ------------------------------
private:

    static void _displayWelcomeMessage();

    static TestParseResult _parseTestInput(TestFunc &out_testFunc);

    // ------------------------------
    // Class fields
    // ------------------------------

    CpuCore *m_cpuCore{};
};


#endif //SRC_TESTRUNNER_CUH
