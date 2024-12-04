#ifndef SRC_PROGRESSBAR_CUH
#define SRC_PROGRESSBAR_CUH

#include <iostream>
#include <string>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

class ProgressBar {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    ProgressBar(__uint32_t total, __uint32_t width) : m_total(total), m_current(0), m_width(width),
                                                      m_numCharacters(0) {
#ifdef _WIN32
        _initializeConsole();
#endif
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    void Start() {
        _clearLine();
        _redrawBar();
    }

    void Increment() {
        std::lock_guard<std::mutex> lock(m_mutex);

        ++m_current;

        if (m_current > m_total) {
            throw std::runtime_error("Extended progress bar points!");
        }

        const auto newNumCharacters = static_cast<__uint32_t>((static_cast<double>(m_current) / m_total) * m_width);

        if (newNumCharacters > m_numCharacters) {
            m_numCharacters = newNumCharacters;
            _clearLine();
            _redrawBar();
        }
    }

    void WriteLine(const std::string &line) {
        std::lock_guard<std::mutex> lock(m_mutex);

        _clearLine();
        std::cout << line << std::endl;
        _redrawBar();
    }

    void Break() {
        throw std::runtime_error("NOT IMPLEMENTED!");

        _clearLine();
        _drawCancelled();
    }

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:

    static void _clearLine() {
#ifdef _WIN32
        _clearConsoleLine();
#else
        std::cout << "\033[K\r";
#endif
    }

    void _redrawBar() const {
        const double progress = static_cast<double>(m_current) / static_cast<double>(m_total);

        std::cout << "[";

        for (__uint32_t i = 0; i < m_numCharacters; ++i) {
            std::cout << "#";
        }

        for (__uint32_t i = m_numCharacters; i < m_width; ++i) {
            std::cout << " ";
        }

        std::cout << "] " << static_cast<int>(progress * 100) << "%";
        std::cout.flush();

        if (m_numCharacters == m_width) {
            std::cout << std::endl;
        }
    }

    void _drawCancelled() const {
        _clearLine();
        std::cout << "[Cancelled]" << std::endl;
    }

    // ------------------------------
    // Windows specifics
    // ------------------------------

#ifdef _WIN32
    HANDLE m_consoleHandle;

    void _initializeConsole() {
        m_consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
        if (m_consoleHandle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Unable to get console handle!");
        }

        DWORD consoleMode;
        if (GetConsoleMode(m_consoleHandle, &consoleMode)) {
            consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(m_consoleHandle, consoleMode);
        }
    }

    void _clearConsoleLine() const {
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (!GetConsoleScreenBufferInfo(m_consoleHandle, &csbi)) {
            throw std::runtime_error("Unable to get console buffer info!");
        }

        DWORD written;
        COORD cursorPosition = csbi.dwCursorPosition;
        cursorPosition.X = 0;

        FillConsoleOutputCharacter(m_consoleHandle, ' ', csbi.dwSize.X, cursorPosition, &written);
        SetConsoleCursorPosition(m_consoleHandle, cursorPosition);
    }
#endif

    // ------------------------------
    // Class fields
    // ------------------------------

    __uint32_t m_total;
    __uint32_t m_current;
    __uint32_t m_width;
    __uint32_t m_numCharacters;

    std::mutex m_mutex{};
};

#endif //SRC_PROGRESSBAR_CUH