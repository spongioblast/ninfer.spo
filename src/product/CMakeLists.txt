add_library(ninfer_media_acquire STATIC
  media_acquire/acquire.cpp
)
ninfer_internal_includes(ninfer_media_acquire)
if(TARGET CURL::libcurl)
  target_link_libraries(ninfer_media_acquire PRIVATE CURL::libcurl)
  target_compile_definitions(ninfer_media_acquire PRIVATE NINFER_HAVE_LIBCURL=1)
elseif(TARGET PkgConfig::LIBCURL)
  target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)
  target_compile_definitions(ninfer_media_acquire PRIVATE NINFER_HAVE_LIBCURL=1)
endif()

add_library(ninfer_product_prompt_input STATIC
  prompt_input/prompt_input.cpp
)
ninfer_internal_includes(ninfer_product_prompt_input)
target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire ninfer::json)

add_library(ninfer_product_logging STATIC
  logging/logging.cpp
  logging/pretty_format.cpp
  logging/startup_log.cpp
)
ninfer_internal_includes(ninfer_product_logging)
target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)
