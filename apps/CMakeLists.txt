add_executable(ninfer-spo
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer-spo)
target_link_libraries(ninfer-spo PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-spo-serve serve/main.cpp)
ninfer_internal_includes(ninfer-spo-serve)
target_link_libraries(ninfer-spo-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-spo-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-spo-perplexity)
target_link_libraries(ninfer-spo-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
