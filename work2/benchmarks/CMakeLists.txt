add_executable(work2_benchmarks
  src/matmul_benchmark.cu
)

target_include_directories(work2_benchmarks
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(work2_benchmarks
  PRIVATE work2_core benchmark::benchmark Eigen3::Eigen
)

# Eigen внутри .cu: nvcc требует этот флаг для constexpr-функций Eigen.
target_compile_options(work2_benchmarks
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
