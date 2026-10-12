target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/ffn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/gdn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/mtp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text_mtp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text_verify.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text_layers.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text_prefill.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/draft.cpp"
)
