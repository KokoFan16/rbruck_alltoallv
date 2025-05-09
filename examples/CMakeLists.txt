cmake_minimum_required (VERSION 3.10)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/examples/*.cpp")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-std=c++11")

add_executable(rbruckv ./rbruckv_example.cpp)
add_executable(twolayer ./twolayer_example.cpp)
add_executable(combAlltoallv ./twophaseLayer_example.cpp)
add_executable(MLinputs ./MLinputs_example.cpp)
add_executable(mpich_twolayer_alltoall ./mpich_twolayer_alltoall.cpp)
add_executable(ms_benchmark ./ms_benchmark.cpp)
add_executable(LTRNA ./twophaseLayer_example2.cpp)
add_executable(TTPL ./twophaseLayer_example3.cpp)
add_executable(TLLiner_S2 ./twophaseLayer_example4.cpp)
add_executable(FFT ./FFT_distribution.cpp)
add_executable(otherDis ./other_distributions.cpp)
add_executable(balance ./balanced_benchmark.cpp)
add_executable(gather ./gather_benchmark.cpp)
add_executable(gather_ata ./gather_alltoall.cpp)



target_link_libraries(rbruckv rbruckvLib)
target_link_libraries(twolayer rbruckvLib)
target_link_libraries(combAlltoallv rbruckvLib)
target_link_libraries(MLinputs rbruckvLib)
target_link_libraries(mpich_twolayer_alltoall rbruckvLib)
target_link_libraries(ms_benchmark rbruckvLib)
target_link_libraries(LTRNA rbruckvLib)
target_link_libraries(TTPL rbruckvLib)
target_link_libraries(TLLiner_S2 rbruckvLib)
target_link_libraries(FFT rbruckvLib)
target_link_libraries(otherDis rbruckvLib)

target_link_libraries(balance rbruckvLib)
target_link_libraries(gather rbruckvLib)
target_link_libraries(gather_ata rbruckvLib)
