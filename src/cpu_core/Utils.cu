//
// Created by Jlisowskyy on 05/12/24.
//

#include "Utils.cuh"

#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

void ClearLines(__uint32_t numLines) {
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hConsole == INVALID_HANDLE_VALUE) return;

    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (!GetConsoleScreenBufferInfo(hConsole, &csbi)) return;

    COORD cursorPosition = csbi.dwCursorPosition;
    DWORD charsWritten;

    for (__uint32_t i = 0; i < numLines; ++i) {
        if (cursorPosition.Y > 0) {
            cursorPosition.Y -= 1;

            cursorPosition.X = 0;

            FillConsoleOutputCharacter(hConsole, ' ', csbi.dwSize.X, cursorPosition, &charsWritten);

            SetConsoleCursorPosition(hConsole, cursorPosition);
        }
    }
#else
    for (__uint32_t i = 0; i < numLines; ++i) {
        std::cout << "\033[A";
        std::cout << "\033[K";
    }
    std::cout.flush();
#endif
}

void InitializeConsole() {
#ifdef _WIN32
    m_consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
        if (m_consoleHandle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Unable to get console handle!");
        }

        DWORD consoleMode;
        if (GetConsoleMode(m_consoleHandle, &consoleMode)) {
            consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(m_consoleHandle, consoleMode);
        }
#endif
}

void CleanCurrentLine() {
    std::cout << "\033[K\r";
}

void StrToUpper(std::string &str) {
    for (char &c: str) {
        c = std::toupper(c);
    }
}

void StrToLower(std::string &str) {
    for (char &c: str) {
        c = std::tolower(c);
    }
}
