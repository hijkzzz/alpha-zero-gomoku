cmake_minimum_required(VERSION 3.0.0)
project(alphazero)
set(CMAKE_BUILD_TYPE DEBUG)
enable_testing()

# Libraries
include_directories(${PROJECT_SOURCE_DIR}/src)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_LIB_SRCS)
add_library(alphazero ${DIR_LIB_SRCS})

# Test
add_executable(gomoku_test ${PROJECT_SOURCE_DIR}/test/gomoku_test.cpp)
target_link_libraries(gomoku_test alphazero)
add_test(gomoku_test gomoku_test)

add_executable(mcts_test ${PROJECT_SOURCE_DIR}/test/mcts_test.cpp)
target_link_libraries(mcts_test alphazero)
add_test(mcts_test mcts_test)
