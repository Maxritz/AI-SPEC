# Use a complete ROCm clang install, not a hipcc wrapper with a stale embedded
# path. CMAKE_HIP_COMPILER may be supplied explicitly for Windows TheRock.
include(CheckLanguage)
set(CMAKE_HIP_ARCHITECTURES ${KNJ_GPU_ARCHS})
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
  message(FATAL_ERROR "HIP requested but ROCm clang was not found. Set CMAKE_HIP_COMPILER and CMAKE_HIP_COMPILER_ROCM_ROOT to your complete ROCm installation.")
endif()
enable_language(HIP)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
find_package(hip CONFIG REQUIRED)
