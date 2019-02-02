cmake_minimum_required(VERSION 3.0.0)
project(library)

option(TEST_MODE "test mode" ON)

# find package
find_package(Torch REQUIRED)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

# add source
include_directories(./src)
aux_source_directory(./src SOURCES)

if(TEST_MODE)
    # test submodule
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
    #swig module
    set_source_files_properties(./src/library.i PROPERTIES CPLUSPLUS ON)
    set_source_files_properties(./src/library.i PROPERTIES SWIG_FLAGS "-includeall")
    swig_add_module(library python ./src/library.i ${SOURCES})
    swig_link_libraries(library ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})

endif()
