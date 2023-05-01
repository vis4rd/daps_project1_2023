cmake_minimum_required(VERSION 3.11)

project(daps_project1_2023 CXX C)

set(MPICH_PATH "/opt/nfs/mpich-4.0.1")
set(MPICH_BINARY_PATH "${MPICH_PATH}/bin")
set(MPICH_INCLUDE_PATH "${MPICH_PATH}/include")
set(MPICH_LIBRARY_PATH "${MPICH_PATH}/lib")
set(MPICH_COMPILER "${MPICH_BINARY_PATH}/mpic++")
set(MPICH_RUNNER "${MPICH_BINARY_PATH}/mpiexec")
set(PARALLEL_TARGETS_FILE "${PROJECT_SOURCE_DIR}/build/nodes")
set(PARALLEL_PROCESSES_COUNT 16)

set(CMAKE_VERBOSE_MAKEFILE False)
set(CMAKE_CXX_COMPILER "${MPICH_COMPILER}")

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/bin" OR NOT IS_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
    message(STATUS "Creating binary directory '${PROJECT_SOURCE_DIR}/bin'")
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
endif()

add_executable(daps_project1)
set_target_properties(daps_project1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
target_sources(daps_project1 PRIVATE "${PROJECT_SOURCE_DIR}/main.cpp")
target_compile_definitions(daps_project1 PRIVATE "-DUSE_MPI")
target_compile_features(daps_project1 PRIVATE c_std_23 cxx_std_20)
target_include_directories(daps_project1 PRIVATE "${MPICH_INCLUDE_PATH}")
target_link_directories(daps_project1 PRIVATE "${MPICH_LIBRARY_PATH}")
target_link_libraries(daps_project1 PRIVATE m mpi)

add_custom_target(run ${MPICH_RUNNER} -f nodes -n 16 $<TARGET_FILE:daps_project1>)
add_custom_command(
    TARGET daps_project1 POST_BUILD
    COMMAND /opt/nfs/config/station204_name_list.sh
    ARGS 1 16 > ${PARALLEL_TARGETS_FILE}
    COMMENT "Generating parallel targets file: '${PARALLEL_TARGETS_FILE}'"
)
