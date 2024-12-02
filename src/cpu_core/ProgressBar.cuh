//
// Created by Jlisowskyy on 02/12/24.
//

#ifndef SRC_PROGRESSBAR_CUH
#define SRC_PROGRESSBAR_CUH

#include <iostream>
#include <string>

class ProgressBar {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    ProgressBar(__uint32_t total, __uint32_t width) : m_total(total), m_current(0), m_width(width),
                                                      m_numCharacters(0) {}

    // ------------------------------
    // Class interaction
    // ------------------------------

    void Start() {
        _clearLine();
        _redrawBar();
    }

    void Increment() {
        ++m_current;

        if (m_current > m_total) {
            throw std::runtime_error("Extended progress bar points!");
        }

        const __uint32_t newNumCharacters = m_current / m_total;

        if (newNumCharacters > m_numCharacters) {
            m_numCharacters = newNumCharacters;
            _clearLine();
            _redrawBar();
        }
    }

    void WriteLine(const std::string &line) {
        _clearLine();
        std::cout << line << std::endl;
        _redrawBar();
    }

    void Break() {
        _clearLine();
        _drawCancelled();
    }

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:

    static void _clearLine() {
        std::cout << "\033[K";
    }

    void _redrawBar() const {
        const double progress = static_cast<double>(m_current) / static_cast<double>(m_total);

        std::cout << "\r[";

        for (__uint32_t i = 0; i < m_numCharacters; ++i) {
            std::cout << "#";
        }

        for (__uint32_t i = m_numCharacters; i < m_width; ++i) {
            std::cout << " ";
        }

        std::cout << "] " << static_cast<int>(progress * 100) << "%";
        std::cout.flush();
    }

    void _drawCancelled() const {
        throw std::runtime_error("NOT IMPLEMENTED");
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    __uint32_t m_total;
    __uint32_t m_current;
    __uint32_t m_width;
    __uint32_t m_numCharacters;
};

#endif //SRC_PROGRESSBAR_CUH
