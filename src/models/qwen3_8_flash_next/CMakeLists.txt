target_sources(ninfer_model_runtime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/activation_control.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/chat_template.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/config_validation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/output_decoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/output_session.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/prompt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/media_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/processor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/tokenizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frontend/tool_call_parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/ple_table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/prefix_identity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/prefill_graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/text_context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/text_layers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/text_mtp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/text_verify.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/vision_context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/prefill.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/decode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/mtp_decode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/execution/target_verify_accept.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/program/p3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/program/p4_mtp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/program/p5_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/program/p_vision.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/program/p_grammar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/program/p_score.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/program/v3_program.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/layouts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/sequence_plan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/decoder_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/state_image.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/round_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision/control.cpp)

target_link_libraries(ninfer_model_runtime PRIVATE CUDA::cuda_driver)

target_include_directories(ninfer_model_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
