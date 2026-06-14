cmake_minimum_required(VERSION 3.16)

project(dlx_plus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_CPSAT "Build the OR-Tools CP-SAT executable" OFF)

if(BUILD_CPSAT)
    find_package(ortools CONFIG REQUIRED)

    add_executable(
        cpsat_experiment
        dlx_experiment.cpp
        cpsat_spk.cpp
    )
    target_compile_definitions(cpsat_experiment PRIVATE USE_CPSAT)
    target_link_libraries(cpsat_experiment PRIVATE ortools::ortools)

    set_target_properties(
        cpsat_experiment
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()
