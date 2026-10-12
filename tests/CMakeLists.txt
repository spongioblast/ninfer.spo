find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(program/tests.cmake)
include(models/qwen3_5/tests.cmake)
include(models/qwen3_8_flash_next/tests.cmake)
include(ops/tests.cmake)

# The oracles pipe UTF-8 text to the probes; Windows Python defaults to the ANSI code page.
set_property(TEST ninfer_chat_templates_test ninfer_regex_choice_oracle_test
  ninfer_json_schema_oracle_test ninfer_qwen3_5_tool_schema_oracle_test
  APPEND PROPERTY ENVIRONMENT "PYTHONUTF8=1")
