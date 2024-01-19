cmake_minimum_required (VERSION 2.8)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

project (RBURKV)
set (CMAKE_CXX_STANDARD 11)

message("Configuring MPR with MPI support")
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/examples")

add_subdirectory(src)
add_subdirectory(examples)

