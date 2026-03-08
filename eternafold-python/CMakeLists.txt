cmake_minimum_required(VERSION 3.15)
project(eternafold LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Fetch EternaFold C++ source at configure time ---
include(FetchContent)
FetchContent_Declare(
    eternafold_upstream
    GIT_REPOSITORY https://github.com/eternagame/EternaFold.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eternafold_upstream)

set(ETERNAFOLD_SRC_DIR "${eternafold_upstream_SOURCE_DIR}/src")
set(ETERNAFOLD_PARAMS_DIR "${eternafold_upstream_SOURCE_DIR}/parameters")

# Compile flags matching upstream Makefile
add_compile_definitions(EVIDENCE_SR EVIDENCE_PARS NDEBUG)

if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O3 -pipe -Wall -Wno-unused-variable -Wno-sign-compare)
    # Only add SSE flags on x86/x64, not on ARM
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
        add_compile_options(-mfpmath=sse -msse -msse2)
    endif()
endif()

# --- Static library of core EternaFold C++ sources ---
add_library(eternafold_core STATIC
    "${ETERNAFOLD_SRC_DIR}/FileDescription.cpp"
    "${ETERNAFOLD_SRC_DIR}/Options.cpp"
    "${ETERNAFOLD_SRC_DIR}/SStruct.cpp"
    "${ETERNAFOLD_SRC_DIR}/Utilities.cpp"
    "${ETERNAFOLD_SRC_DIR}/contrafold_api.cpp"
)
target_include_directories(eternafold_core PUBLIC "${ETERNAFOLD_SRC_DIR}")
target_link_libraries(eternafold_core PUBLIC m)

# --- CLI binary (contrafold) for Arnie compatibility ---
add_executable(contrafold "${ETERNAFOLD_SRC_DIR}/Contrafold.cpp")
target_include_directories(contrafold PRIVATE "${ETERNAFOLD_SRC_DIR}")
target_link_libraries(contrafold PRIVATE eternafold_core)
install(TARGETS contrafold DESTINATION eternafold/bin)

# --- pybind11 extension module ---
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE bindings/pybind_eternafold.cpp)
target_link_libraries(_core PRIVATE eternafold_core)
target_include_directories(_core PRIVATE "${ETERNAFOLD_SRC_DIR}")
install(TARGETS _core DESTINATION eternafold)

# --- Install parameter files as package data ---
install(
    DIRECTORY "${ETERNAFOLD_PARAMS_DIR}/"
    DESTINATION eternafold/parameters
    FILES_MATCHING
    PATTERN "*.v1"
    PATTERN "LICENSE"
    PATTERN "README.md"
)
