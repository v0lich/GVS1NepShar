add_executable(work2_tests
  src/matmul_test.cu
)

target_include_directories(work2_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(work2_tests
  PRIVATE work2_core GTest::gtest_main Eigen3::Eigen
)

# Eigen внутри .cu: nvcc требует этот флаг для constexpr-функций Eigen.
target_compile_options(work2_tests
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

include(GoogleTest)
gtest_discover_tests(work2_tests)
