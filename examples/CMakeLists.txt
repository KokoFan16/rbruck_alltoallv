cmake_minimum_required (VERSION 3.5)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/examples/*.cpp")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-std=c++11")

add_executable(rbruckv ./rbruckv_example.cpp)
add_executable(twolayer ./twolayer_example.cpp)
add_executable(combAlltoallv ./twophaseLayer_example.cpp)
add_executable(MLinputs ./MLinputs_example.cpp)
add_executable(mpich_twolayer_alltoall ./mpich_twolayer_alltoall.cpp)
add_executable(ms_benchmark ./ms_benchmark.cpp)


target_link_libraries(rbruckv rbruckvLib)
target_link_libraries(twolayer rbruckvLib)
target_link_libraries(combAlltoallv rbruckvLib)
target_link_libraries(MLinputs rbruckvLib)
target_link_libraries(mpich_twolayer_alltoall rbruckvLib)
target_link_libraries(ms_benchmark rbruckvLib)