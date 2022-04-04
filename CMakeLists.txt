cmake_minimum_required(VERSION 3.8.0)
project(library)

# option
option(WRAP_LIB "wrap library" ON)
option(UNIT_TEST "unit test" OFF)

# gcc
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_STANDARD 14)
endif()

# find torch
find_package(Torch REQUIRED)

# find swig
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# find python
find_package(PythonLibs 3.6 REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

# add sources
include_directories(./src)
aux_source_directory(./src SOURCES)

# swig
if(WRAP_LIB)
    set_property(SOURCE ./src/library.i PROPERTY CPLUSPLUS ON)
    swig_add_library(library TYPE SHARED LANGUAGE python SOURCES ./src/library.i ${SOURCES})
    swig_link_libraries(library ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})
endif()

# unit test
if(UNIT_TEST)
    add_library(test_lib ${SOURCES})
    target_link_libraries(test_lib ${TORCH_LIBRARIES})

    add_executable(thread_pool_test ./test/thread_pool_test.cpp)
    target_link_libraries(thread_pool_test test_lib)

    add_executable(gomoku_test ./test/gomoku_test.cpp)
    target_link_libraries(gomoku_test test_lib)

    add_executable(libtorch_test ./test/libtorch_test.cpp)
    target_link_libraries(libtorch_test test_lib)

    add_executable(mcts_test ./test/mcts_test.cpp)
    target_link_libraries(mcts_test test_lib)
endif()
