cmake_minimum_required(VERSION 3.14)
project(variationaLLL C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -DNDEBUG")

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FPLLL REQUIRED fplll)
pkg_check_modules(GMP   REQUIRED gmp)
pkg_check_modules(MPFR  REQUIRED mpfr)

add_executable(gdlll_benchmark gdlll_benchmark.cpp)

target_include_directories(gdlll_benchmark PRIVATE
    ${FPLLL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS} ${MPFR_INCLUDE_DIRS})
target_link_directories(gdlll_benchmark PRIVATE
    ${FPLLL_LIBRARY_DIRS} ${GMP_LIBRARY_DIRS} ${MPFR_LIBRARY_DIRS})
target_link_libraries(gdlll_benchmark
    ${FPLLL_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(gdlll_benchmark PRIVATE -Wall -Wextra -Wno-unused-parameter)
