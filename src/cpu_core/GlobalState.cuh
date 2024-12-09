//
// Created by Jlisowskyy on 09/12/24.
//

#ifndef GLOBALSTATE_HPP
#define GLOBALSTATE_HPP

#include <unordered_map>
#include <string>
#include <functional>

struct GlobalState {
    bool WriteExtensiveInfo{};
    bool WriteDotFiles{};
    bool WriteTimes{};
    bool RunTestsOnStart{};
    std::string TestName{};
};

extern GlobalState g_GlobalState;
extern std::unordered_map<std::string, std::function<void()> > g_GlobalStateCommands;

void DisplayHelp();

#endif //GLOBALSTATE_HPP
