//
// Created by Jlisowskyy on 09/12/24.
//

#include "GlobalState.cuh"
#include "../tests/CudaTests.cuh"

GlobalState g_GlobalState{};

std::unordered_map<std::string, std::function<void ()> > g_GlobalStateCommands = [
        ]()-> std::unordered_map<std::string, std::function<void ()> > {
            std::unordered_map<std::string, std::function<void ()> > rv{
                {"write_ext_info", []() { g_GlobalState.WriteExtensiveInfo = true; }},
                {"write_dot_files", []() { g_GlobalState.WriteDotFiles = true; }},
                {"write_times", []() { g_GlobalState.WriteTimes = true; }},
            };

            for (const auto &[testName, _]: g_CudaTestsMap) {
                rv[testName] = [testName]() {
                    g_GlobalState.RunTestsOnStart = true;
                    g_GlobalState.TestName = testName;
                };
            }

            return rv;
        }();