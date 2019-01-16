cmake_minimum_required(VERSION 3.0.0)
project(alphazero VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Release)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include(${PROJECT_SOURCE_DIR})
aux_source_directory(${PROJECT_SOURCE_DIR} DIR_LIB_SRCS)
add_library(alphazero ${DIR_LIB_SRCS})