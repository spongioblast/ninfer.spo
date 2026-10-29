target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/decode/sparse_moe_decode_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/decode/sparse_moe_decode_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/sparse_moe_prefill_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/sparse_moe_prefill_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/sparse_moe_small_t_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/sparse_moe_small_t_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/sparse_moe.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/flash_next/flash_next_moe.cu"
  "${CMAKE_CURRENT_LIST_DIR}/flash_next/cpu_tier.cpp"
)
if(MSVC)
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/flash_next/cpu_tier.cpp"
    PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/flash_next/cpu_tier.cpp"
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni;-mavx2")
endif()
