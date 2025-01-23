cmake_minimum_required (VERSION 3.5)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/examples/*.cpp")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-std=c++11")

aux_source_directory(. RBURK_SOURCES)
add_library(benchLib ${RBURK_SOURCES})

#add_executable(merge_benckmark ./merge_benchmark.cpp)
#add_executable(split_benchmark ./split_benchmark.cpp)
#add_executable(simple_merge_benchmark ./simple_merge_benchmark.cpp)
add_executable(intra_node_comm ./intra_node_comm.cpp)

#target_link_libraries(split_benchmark rbruckvLib benchLib)
#target_link_libraries(merge_benckmark rbruckvLib benchLib)
#target_link_libraries(simple_merge_benchmark rbruckvLib benchLib)
target_link_libraries(intra_node_comm rbruckvLib benchLib)
