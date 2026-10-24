ninfer_add_op_test(ninfer_gemma4_text_forward_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_text_forward.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_op_test(ninfer_gemma4_prefill_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefill.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_op_test(ninfer_gemma4_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_op_test(ninfer_gemma4_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_image.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_op_test(ninfer_gemma4_vision_encode_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_encode.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_op_test(ninfer_gemma4_france_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_france.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_op_test(ninfer_gemma4_parity_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_parity_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops ninfer::json)

ninfer_add_op_test(ninfer_gemma4_draft_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_draft_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops ninfer::json)

ninfer_add_op_test(ninfer_gemma4_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_op_test(ninfer_gemma4_checkpoints_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_checkpoints_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_gemma4_checkpoints_real_test
  ninfer_gemma4_france_real_test
  ninfer_gemma4_parity_real_test
  ninfer_gemma4_draft_real_test
  ninfer_gemma4_engine_real_test
  ninfer_gemma4_vision_encode_real_test
  PROPERTIES RUN_SERIAL TRUE LABELS "gpu;real")

ninfer_add_op_test(ninfer_gemma4_tokenizer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tokenizer.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_op_test(ninfer_gemma4_chat_template_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chat_template.cpp"
  LIBRARIES ninfer_jinja ninfer::json)

ninfer_add_op_test(ninfer_gemma4_frontend_fixtures_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_fixtures.cpp"
  LIBRARIES ninfer_model_runtime ninfer::json)
target_compile_definitions(ninfer_gemma4_frontend_fixtures_test PRIVATE
  NINFER_GEMMA4_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/fixtures")

ninfer_add_op_test(ninfer_gemma4_tool_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_parser.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_op_test(ninfer_gemma4_prefix_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_store.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_op_test(ninfer_gemma4_top_tokens_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_top_tokens.cpp"
  LIBRARIES ninfer_model_runtime)
