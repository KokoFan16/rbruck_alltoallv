cmake_minimum_required (VERSION 2.8)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/examples/*.cpp")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-std=c++11")

add_executable(rbruckv ./rbruckv_example.cpp)
add_executable(twolayer ./twolayer_example.cpp)

target_link_libraries(rbruckv rbruckvLib)
target_link_libraries(twolayer rbruckvLib)