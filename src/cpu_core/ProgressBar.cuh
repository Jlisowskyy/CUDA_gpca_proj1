#ifndef SRC_PROGRESSBAR_CUH
#define SRC_PROGRESSBAR_CUH

#include <iostream>
#include <string>
#include <mutex>

#include "Utils.cuh"

/**
 * @brief Provides a thread-safe console progress bar for tracking task completion
 *
 * Displays progress visually with percentage and dynamic bar length
 */
class ProgressBar {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    /**
     * @brief Construct a progress bar with total steps and display width
     *
     * @param total Total number of steps in the task
     * @param width Width of the progress bar in console characters
     */
    ProgressBar(const uint32_t total, const uint32_t width) : m_total(total), m_current(0), m_width(width),
                                                      m_numCharacters(0) {}

    // ------------------------------
    // Class interaction
    // ------------------------------

    /**
     * @brief Initiate progress bar display
     *
     * @note from this moment all writes to stdout should be done through progress bar
     */
    void Start() const {
        _clearLine();
        _redrawBar();
    }

    /**
     * @brief Increment progress, updating display if needed
     *
     * @param increment Number of steps to advance (default 1)
     */
    void Increment(const uint32_t increment = 1) {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_current += increment;

        if (m_current > m_total) {
            throw std::runtime_error("Extended progress bar points!");
        }

        const auto newNumCharacters = static_cast<uint32_t>((static_cast<double>(m_current) / m_total) * m_width);

        if (newNumCharacters > m_numCharacters) {
            m_numCharacters = newNumCharacters;
            _clearLine();
            _redrawBar();
        }
    }

    /**
     * @brief Write a line of text while preserving progress bar
     *
     * @param line Text to display
     */
    void WriteLine(const std::string &line) {
        std::lock_guard<std::mutex> lock(m_mutex);

        _clearLine();
        std::cout << line << std::endl;
        _redrawBar();
    }

    /**
     * @brief Attempt to cancel progress bar (currently not implemented)
     */
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

        for (uint32_t i = 0; i < m_numCharacters; ++i) {
            std::cout << "#";
        }

        for (uint32_t i = m_numCharacters; i < m_width; ++i) {
            std::cout << " ";
        }

        std::cout << "] " << static_cast<int>(progress * 100) << "%";
        std::cout << std::endl;
    }

    static void _drawCancelled() {
        _clearLine();
        std::cout << "[Cancelled]" << std::endl;
    }

    // ------------------------------
    // Windows specifics
    // ------------------------------

    // ------------------------------
    // Class fields
    // ------------------------------

    uint32_t m_total;
    uint32_t m_current;
    uint32_t m_width;
    uint32_t m_numCharacters;

    std::mutex m_mutex{};
};

#endif //SRC_PROGRESSBAR_CUH