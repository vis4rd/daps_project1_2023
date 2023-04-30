cmake_minimum_required(VERSION 3.11)

project(daps_project1_2023 CXX C)

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/bin" OR NOT IS_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
    message(STATUS "Creating binary directory '${PROJECT_SOURCE_DIR}/bin'")
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
endif()

add_library(daps_project1)
set_target_properties(daps_project1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
target_sources(daps_project1 PRIVATE "${PROJECT_SOURCE_DIR}/main.c")
target_compile_definitions(daps_project1 PRIVATE "-DUSE_MPI")
target_compile_features(daps_project1 PRIVATE c_std_23 cxx_std_20)
target_include_directories(daps_project1 PRIVATE "/opt/nfs/mpich-4.0.1/include")
target_link_directories(daps_project1 PRIVATE "/opt/nfs/mpich-4.0.1/lib")
target_link_libraries(daps_project1 PRIVATE m mpi)
