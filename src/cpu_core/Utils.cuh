//
// Created by Jlisowskyy on 05/12/24.
//

#ifndef SRC_UTILS_CUH
#define SRC_UTILS_CUH

#include <string>

/**
 * @brief Initialize console with enhanced processing capabilities
 *
 * Specifically configures Windows console for better terminal support
 */
void InitializeConsole();

/**
 * @brief Clear specified number of lines in console
 *
 * Works cross-platform (Windows and Unix-like systems)
 *
 * @param numLines Number of lines to clear
 */
void ClearLines(uint32_t numLines);

/**
 * @brief Clear current line in console
 *
 * Resets cursor to line start and clears line content
 */
void CleanCurrentLine();

/**
 * @brief Convert string to lowercase
 *
 * @param str Reference to string to be modified
 */
void StrToLower(std::string& str);

/**
 * @brief Convert string to uppercase
 *
 * @param str Reference to string to be modified
 */
void StrToUpper(std::string& str);

#endif //SRC_UTILS_CUH
