#include <iostream>
#include "utilities/BitOperations.hpp"
#include "move_gen/table_based/base_tables/BaseMoveHashMap.hpp"
#include "move_gen/table_based/base_tables/FancyMagicRookMap.hpp"
#include "move_gen/table_based/Board.hpp"
#include "move_gen/table_based/MoveGenerationUtils.hpp"
#include "move_gen/table_based/base_tables/RookMapGenerator.hpp"
#include "move_gen/table_based/base_tables/BishopMapGenerator.hpp"
#include "move_gen/table_based/base_tables/FancyMagicBishopMap.hpp"
#include "move_gen/table_based/tables/BishopMap.hpp"
#include "move_gen/table_based/tables/RookMap.hpp"
#include "move_gen/table_based/tables/KingMap.hpp"
#include "move_gen/table_based/tables/WhitePawnMap.hpp"
#include "move_gen/table_based/tables/BlackPawnMap.hpp"
#include "move_gen/table_based/tables/QueenMap.hpp"
#include "move_gen/table_based/Move.hpp"
#include "move_gen/table_based/ChessMechanics.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
