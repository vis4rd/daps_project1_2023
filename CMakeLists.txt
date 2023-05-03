cmake_minimum_required(VERSION 3.11)

project(daps_project1_2023 CXX C)

option(ENABLE_LOGGING "Print debug logs" False)
option(ENABLE_ASAN "Enable address sanitizer" False)

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

add_custom_command(
    OUTPUT ${PARALLEL_TARGETS_FILE}
    COMMAND /opt/nfs/config/station204_name_list.sh
    ARGS 1 16 > ${PARALLEL_TARGETS_FILE}
    COMMENT "Generating parallel targets file: '${PARALLEL_TARGETS_FILE}'"
)
add_library(parallel_nodes OBJECT ${PARALLEL_TARGETS_FILE})

###

add_executable(fft_parallel)
set_target_properties(fft_parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
target_sources(fft_parallel PRIVATE "${PROJECT_SOURCE_DIR}/fft_parallel.cpp")
target_compile_definitions(fft_parallel PRIVATE
    USE_MPI
    $<$<BOOL:${ENABLE_LOGGING}>:ENABLE_LOGGING>
)
target_compile_features(fft_parallel PRIVATE c_std_23 cxx_std_20)
target_compile_options(fft_parallel PRIVATE $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>)
target_include_directories(fft_parallel PRIVATE "${MPICH_INCLUDE_PATH}")
target_link_directories(fft_parallel PRIVATE "${MPICH_LIBRARY_PATH}")
target_link_libraries(fft_parallel PRIVATE m mpi)
target_link_options(fft_parallel PRIVATE $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>)

add_custom_target(run
    ${MPICH_RUNNER} -f ${PARALLEL_TARGETS_FILE}
                    -n ${PARALLEL_PROCESSES_COUNT}
                    $<TARGET_FILE:fft_parallel>
    DEPENDS $<TARGET_FILE:fft_parallel>
            parallel_nodes
)

###

add_executable(fft_sequential)
set_target_properties(fft_sequential PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
target_sources(fft_sequential PRIVATE "${PROJECT_SOURCE_DIR}/fft_sequential.c")
target_compile_features(fft_sequential PRIVATE c_std_23 cxx_std_20)
target_link_libraries(fft_sequential PRIVATE m)

add_custom_target(sequential $<TARGET_FILE:fft_sequential>
    DEPENDS $<TARGET_FILE:fft_sequential>
)
