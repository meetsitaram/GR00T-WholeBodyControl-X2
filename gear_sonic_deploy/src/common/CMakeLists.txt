# =============================================================================
# sonic_common -- robot-agnostic deploy-time utilities shared across robots.
# =============================================================================
#
# Phase 2 (current) deliberately does NOT relocate source files. The curated
# "portable" headers still live under
# `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/`; this INTERFACE
# library simply re-exports their parent directory under the `sonic_common`
# namespace so downstream consumers (Phase 3 X2 deploy package, future
# robots, etc.) can write
#
#     target_link_libraries(my_target PRIVATE sonic_common)
#     #include "fk.hpp"          // resolved via sonic_common include path
#
# without depending on the literal `g1/g1_deploy_onnx_ref/include/` path.
# When a follow-up cleanup phase (post X2 bring-up) physically moves these
# headers into `src/common/include/`, the consumer code does NOT need to
# change -- only this CMakeLists.txt does.
#
# IMPORTANT: g1_deploy_onnx_ref does NOT consume sonic_common. The G1 build
# remains unchanged. See README.md for the audit, the migration plan, and
# the explicit list of headers considered portable.
#
# This subdirectory is intentionally NOT yet wired into ../CMakeLists.txt.
# Phase 3 (X2 deploy package) will add `add_subdirectory(common)` together
# with `add_subdirectory(x2)`.

cmake_minimum_required(VERSION 3.16)

add_library(sonic_common INTERFACE)

target_include_directories(sonic_common
  INTERFACE
    # New code added directly to src/common/ lands here.
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # Curated portable headers still living in the G1 tree.
    # See README.md "Curated portable subset" for the explicit list and
    # for the rule of "do not add new files here -- add them under
    # src/common/include/sonic_common/ instead."
    ${CMAKE_CURRENT_SOURCE_DIR}/../g1/g1_deploy_onnx_ref/include
)

target_compile_features(sonic_common INTERFACE cxx_std_20)
