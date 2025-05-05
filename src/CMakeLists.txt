cmake_minimum_required (VERSION 3.10)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/*.cpp")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX )

aux_source_directory(. RBURK_SOURCES)
add_library(rbruckvLib ${RBURK_SOURCES})

target_link_libraries(rbruckvLib ${MPI_C_LIBRARIES})
