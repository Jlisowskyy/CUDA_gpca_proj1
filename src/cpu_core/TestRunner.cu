//
// Created by Jlisowskyy on 16/11/24.
//

#include "TestRunner.cuh"
#include "CpuCore.cuh"

#include <cassert>
#include <iostream>
#include <string_view>

TestRunner::TestRunner(CpuCore *cpuCore) : m_cpuCore(cpuCore) {
    assert(m_cpuCore != nullptr && "CpuCore is nullptr");
}

TestRunner::~TestRunner() = default;

void TestRunner::runTests() {
    TestParseResult parseResult = TestParseResult::SUCCESS;

    while (parseResult != TestParseResult::EXIT) {
        TestFunc testFunc{};

        _displayWelcomeMessage();
        while ((parseResult = _parseTestInput(testFunc)) == TestParseResult::FAILURE) {
            std::cout << "Invalid input, please try again" << std::endl;
        }

        if (parseResult == TestParseResult::SUCCESS) {
            testFunc(m_cpuCore->getDeviceThreads(), m_cpuCore->getDeviceProps());
        }
    }
}

void TestRunner::_displayWelcomeMessage() {
    static constexpr std::string_view MSG = R"(
        Welcome to the Checkmate-Chariot-CUDA test mode!

        To pick test please use TEST CODE from the list below.
        There are several tests available for you to run:
)";

    std::cout << MSG << std::endl;

    for (const auto &test: g_CudaTestsMap) {
        const auto &[name, desc, _] = test.second;

        std::cout << "Code: " << test.first << std::endl;
        std::cout << "Test: " << name << std::endl;
        std::cout << "Description: " << desc << std::endl;
        std::cout << "-----------------------------" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Please enter the name of the test you would like to run, or 'exit' to quit" << std::endl;
}

TestRunner::TestParseResult TestRunner::_parseTestInput(TestFunc &out_testFunc) {
    std::cout << "Enter test code: ";

    std::string input;
    std::getline(std::cin, input);

    if (input == "exit") {
        return TestRunner::TestParseResult::EXIT;
    }

    if (g_CudaTestsMap.find(input) == g_CudaTestsMap.end()) {
        return TestRunner::TestParseResult::FAILURE;
    }

    const auto &[_, __, testFunc] = g_CudaTestsMap.at(input);
    out_testFunc = testFunc;
    return TestRunner::TestParseResult::SUCCESS;
}
