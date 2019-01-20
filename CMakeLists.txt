cmake_minimum_required(VERSION 3.0.0)
project(lib)
set(CMAKE_BUILD_TYPE DEBUG)

# python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# library
include_directories(${PROJECT_SOURCE_DIR}/src)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_LIB_SRCS)
add_library(lib ${DIR_LIB_SRCS})
target_link_libraries(lib ${PYTHON_LIBRARIES})

# test
enable_testing()

# gomoku
add_executable(gomoku_test ${PROJECT_SOURCE_DIR}/test/gomoku_test.cpp)
target_link_libraries(gomoku_test lib)
add_test(gomoku_test gomoku_test)

# thread pool
add_executable(thread_pool_test ${PROJECT_SOURCE_DIR}/test/thread_pool_test.cpp)
add_test(thread_pool_test thread_pool_test)

# mcts
add_executable(mcts_test ${PROJECT_SOURCE_DIR}/test/mcts_test.cpp)
target_link_libraries(mcts_test lib)
add_test(mcts_test mcts_test)
