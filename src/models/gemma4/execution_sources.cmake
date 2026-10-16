target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/draft_forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/projection.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text_forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tokenizer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_parser.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/top_tokens.cpp"
)
