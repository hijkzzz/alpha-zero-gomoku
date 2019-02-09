cmake_minimum_required(VERSION 3.0.0)
project(library)

option(TEST_MODE "test mode" OFF)

# add source directory
include_directories(./src)
aux_source_directory(./src SOURCES)

# find torch
find_package(Torch REQUIRED)

if(TEST_MODE)
    # add test library
    add_library(test ${SOURCES})
    target_link_libraries(test ${TORCH_LIBRARIES})

    add_executable(thread_pool_test ./test/thread_pool_test.cpp)
    target_link_libraries(thread_pool_test test)

    add_executable(gomoku_test ./test/gomoku_test.cpp)
    target_link_libraries(gomoku_test test)

    add_executable(libtorch_test ./test/libtorch_test.cpp)
    target_link_libraries(libtorch_test test)

    add_executable(mcts_test ./test/mcts_test.cpp)
    target_link_libraries(mcts_test test)

else()
    # find swig
    find_package(SWIG REQUIRED)
    include(${SWIG_USE_FILE})

    # find python
    find_package(PythonLibs)
    include_directories(${PYTHON_INCLUDE_PATH})

    # swig add module
    set_property(SOURCE ./src/library.i PROPERTY CPLUSPLUS ON)
    swig_add_library(library TYPE SHARED LANGUAGE python SOURCES ./src/library.i ${SOURCES})
    swig_link_libraries(library ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})

endif()
