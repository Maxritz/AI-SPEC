cmake_minimum_required(VERSION 3.16)
project(kanjoos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Host-only substrate (GGUF reader, model index). No GPU/HIP dependency:
# see docs/07-build-platforms.md section 2.
add_library(knj_substrate STATIC
  src/util/sha256.cpp
  src/gguf/gguf_reader.cpp
  src/gguf/model_index.cpp)
target_include_directories(knj_substrate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(MSVC)
  target_compile_options(knj_substrate PRIVATE /W4)
else()
  target_compile_options(knj_substrate PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(kanjoos-inspect src/tools/kanjoos_inspect.cpp)
target_link_libraries(kanjoos-inspect PRIVATE knj_substrate)

enable_testing()
add_executable(test_substrate tests/unit/test_substrate.cpp)
target_link_libraries(test_substrate PRIVATE knj_substrate)
target_include_directories(test_substrate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit)
add_test(NAME substrate COMMAND test_substrate)
