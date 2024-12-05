#ifndef SRC_PROGRESSBAR_CUH
#define SRC_PROGRESSBAR_CUH

#include <iostream>
#include <string>
#include <mutex>

#include "Utils.cuh"

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
        ClearLines(1);
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
        std::cout << std::endl;
    }

    void _drawCancelled() const {
        _clearLine();
        std::cout << "[Cancelled]" << std::endl;
    }

    // ------------------------------
    // Windows specifics
    // ------------------------------

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