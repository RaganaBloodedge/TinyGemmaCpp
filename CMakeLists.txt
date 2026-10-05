cmake_minimum_required(VERSION 3.16)
project(TinyGemmaCpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(tgcore STATIC
  src/config.cc
  src/sbs_reader.cc
  src/weights.cc
  src/matmul.cc
  src/ops.cc
  src/model.cc
  src/threads.cc
)
target_include_directories(tgcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tgcore PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(tgcore PUBLIC TG_HAVE_OPENMP=1)
  message(STATUS "OpenMP: enabled")
else()
  message(STATUS "OpenMP: not found, single-threaded")
endif()

if(MSVC)
  target_compile_options(tgcore PRIVATE /O2 /W4)
else()
  target_compile_options(tgcore PRIVATE -O3 -Wall -Wextra)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" TG_HAS_MARCH_NATIVE)
  if(TG_HAS_MARCH_NATIVE)
    target_compile_options(tgcore PRIVATE -march=native)
  endif()
endif()

add_executable(tinygemma src/main.cc)
target_link_libraries(tinygemma PRIVATE tgcore)

enable_testing()
add_executable(test_ops tests/test_ops.cc)
target_link_libraries(test_ops PRIVATE tgcore)
add_test(NAME ops COMMAND test_ops)

add_executable(test_sbs tests/test_sbs.cc)
target_link_libraries(test_sbs PRIVATE tgcore)
add_test(NAME sbs COMMAND test_sbs)
# test_sbs 需要 .sbs 权重（本仓库不带）。没有权重时它返回 77 表示跳过，
# 而不是把缺文件报成测试失败。
set_tests_properties(sbs PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_matmul tests/test_matmul.cc)
target_link_libraries(test_matmul PRIVATE tgcore)
add_test(NAME matmul COMMAND test_matmul)

add_executable(test_threads tests/test_threads.cc)
target_link_libraries(test_threads PRIVATE tgcore)
add_test(NAME threads COMMAND test_threads)

# 内核微基准：不需要权重文件，秒级出结果
add_executable(bench_kernels tests/bench_kernels.cc)
target_link_libraries(bench_kernels PRIVATE tgcore)

add_executable(bench_gemm_batch tests/bench_gemm_batch.cc)
target_link_libraries(bench_gemm_batch PRIVATE tgcore)
