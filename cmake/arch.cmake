# One source of truth. Never manually define __gfx*__; they are device-pass
# macros supplied by clang. Wave32 is mandatory on BOTH native targets.
foreach(arch IN LISTS KNJ_GPU_ARCHS)
  if(NOT arch STREQUAL "gfx1031" AND NOT arch STREQUAL "gfx1201")
    message(FATAL_ERROR "Supported native targets: gfx1031, gfx1201 (got ${arch})")
  endif()
endforeach()
# Per-device-pass flags, rather than global -Xclang flags accidentally applied
# to the host pass. No dot1-insts override and no wmma-256b-insts.
set(KNJ_HIP_WAVE_FLAGS "SHELL:-Xarch_device -mno-wavefrontsize64")
