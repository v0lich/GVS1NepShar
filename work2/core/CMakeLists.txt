add_library(work2_core
  src/matrix.cu
)

target_include_directories(work2_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(work2_core
  PUBLIC CUDA::cudart
)

target_compile_options(work2_core
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
