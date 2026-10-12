ninfer_add_test(ninfer_program_attention_only_core_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attention_only_core.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_program_attention_only_core_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
