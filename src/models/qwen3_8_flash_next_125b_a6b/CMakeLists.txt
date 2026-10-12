target_sources(ninfer_model_runtime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load_plan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp)

target_include_directories(ninfer_model_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
