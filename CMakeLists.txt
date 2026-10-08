cmake_minimum_required(VERSION 3.16)
project(kanjoos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Host-only layers (no GPU/HIP dependency): docs/07-build-platforms.md section 2.
add_library(knj_substrate STATIC
  src/util/sha256.cpp
  src/gguf/gguf_reader.cpp
  src/gguf/model_index.cpp)
target_include_directories(knj_substrate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(knj_tensor STATIC
  src/tensor/tensor_io.cpp)
target_link_libraries(knj_tensor PUBLIC knj_substrate)

add_library(knj_router STATIC
  src/router/router.cpp)
target_link_libraries(knj_router PUBLIC knj_tensor)

add_library(knj_store STATIC
  src/store/expert_store.cpp)
target_link_libraries(knj_store PUBLIC knj_substrate)

if(MSVC)
  set(KNJ_WARNINGS /W4)
else()
  set(KNJ_WARNINGS -Wall -Wextra -Wpedantic)
endif()
target_compile_options(knj_substrate PRIVATE ${KNJ_WARNINGS})
target_compile_options(knj_store PRIVATE ${KNJ_WARNINGS})
target_compile_options(knj_tensor PRIVATE ${KNJ_WARNINGS})
target_compile_options(knj_router PRIVATE ${KNJ_WARNINGS})

add_executable(kanjoos-inspect src/tools/kanjoos_inspect.cpp)
target_link_libraries(kanjoos-inspect PRIVATE knj_substrate)

add_executable(kanjoos-store src/tools/kanjoos_store.cpp)
target_link_libraries(kanjoos-store PRIVATE knj_store)

enable_testing()
add_executable(test_substrate tests/unit/test_substrate.cpp)
target_link_libraries(test_substrate PRIVATE knj_substrate)
target_include_directories(test_substrate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit)
add_test(NAME substrate COMMAND test_substrate)

add_executable(test_store tests/unit/test_store.cpp)
target_link_libraries(test_store PRIVATE knj_store)
target_include_directories(test_store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit)
add_test(NAME store COMMAND test_store)

add_executable(test_router tests/unit/test_router.cpp)
target_link_libraries(test_router PRIVATE knj_router)
add_test(NAME router COMMAND test_router)
