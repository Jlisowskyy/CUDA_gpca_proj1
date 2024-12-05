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

    /**
     * @brief Constructs TestRunner with a CPU core context
     *
     * @param cpuCore Pointer to CpuCore for device-specific test execution
     */
    explicit TestRunner(CpuCore *cpuCore);

    ~TestRunner();

    // ------------------------------
    // Class interaction
    // ------------------------------

    /**
     * @brief Runs interactive test selection and execution loop
     *
     * Displays available tests, prompts for test selection,
     * and executes chosen tests until user exits
     */
    void runTests();

    // ------------------------------
    // Class private methods
    // ------------------------------
private:

    /**
     * @brief Displays welcome message and list of available tests
     *
     * Shows test codes, names, and descriptions to guide user selection
     */
    static void _displayWelcomeMessage();

    /**
     * @brief Parses user input and selects corresponding test function
     *
     * @param out_testFunc Reference to store selected test function
     * @return TestParseResult Indicates input validation status
     */
    static TestParseResult _parseTestInput(TestFunc &out_testFunc);

    // ------------------------------
    // Class fields
    // ------------------------------

    CpuCore *m_cpuCore{};
};


#endif //SRC_TESTRUNNER_CUH
