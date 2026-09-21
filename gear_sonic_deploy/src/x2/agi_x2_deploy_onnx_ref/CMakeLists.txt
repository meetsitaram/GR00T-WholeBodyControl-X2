# =============================================================================
# agi_x2_deploy_onnx_ref -- AgiBot X2 Ultra ONNX deploy harness (Phase 3).
#
# Build steps on the X2 dev box:
#   colcon build --packages-select aimdk_msgs            # from thirdparty/aimdk_msgs (AimDK SDK)
#   colcon build --packages-select agi_x2_deploy_onnx_ref \
#       --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
#
# Standalone CMake (offline syntax check, no ROS 2 install present):
#   cmake -S . -B build -DAGI_X2_OFFLINE_SYNTAX_CHECK=ON
#
# This CMakeLists deliberately does NOT bundle the AimDK message sources.
# They come from AgiBot's AimDK SDK (gear_sonic_deploy/thirdparty/aimdk_msgs/
# or PC2's system install) and must be built first (see README.md
# "Build prerequisites").
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(agi_x2_deploy_onnx_ref VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Pull in the robot-agnostic shared utilities (sonic_common). The Phase 2
# README explicitly designates this as the consumer hook for X2 -- the
# add_subdirectory call here is what physically activates sonic_common.
# -----------------------------------------------------------------------------
if(NOT TARGET sonic_common)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../common
                   ${CMAKE_BINARY_DIR}/sonic_common_build)
endif()

# -----------------------------------------------------------------------------
# Offline-syntax-check mode lets us run `cmake -S . -B build` on a workstation
# that does NOT have ROS 2 / aimdk_msgs / onnxruntime installed. It builds
# only the obs-builder unit tests (which depend on nothing but the standard
# library + sonic_common headers) so we can catch most C++ regressions in CI
# without spinning up the full deploy environment.
# -----------------------------------------------------------------------------
option(AGI_X2_OFFLINE_SYNTAX_CHECK
       "Skip ROS 2 / onnxruntime targets (header + unit-test only)" OFF)

if(AGI_X2_OFFLINE_SYNTAX_CHECK)
  message(STATUS "agi_x2_deploy_onnx_ref: OFFLINE syntax-check mode "
                 "(skipping ROS 2 deploy binary)")

  # ZMQ + nlohmann_json find -- only required for the ZMQ helpers, but we
  # surface them in the offline mode too so the M2 wiring (subscriber /
  # pose source / debug publisher) can be syntax-checked on a workstation
  # that doesn't have ROS 2 available. Both are header-only on Debian
  # (nlohmann_json) or come with a libzmq dev package.
  find_path(ZMQ_INCLUDE_DIR zmq.hpp
    PATHS /usr/include /usr/local/include)
  find_library(ZMQ_LIBRARY zmq
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
  find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    PATHS /usr/include /usr/local/include)
  if(NOT ZMQ_INCLUDE_DIR OR NOT ZMQ_LIBRARY OR NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(WARNING "agi_x2: skipping ZMQ helpers in offline mode (missing zmq/nlohmann_json)")
    set(AGI_X2_BUILD_ZMQ_HELPERS OFF)
  else()
    set(AGI_X2_BUILD_ZMQ_HELPERS ON)
  endif()

  add_library(agi_x2_obs STATIC
    src/proprioception_buffer.cpp
    src/tokenizer_obs.cpp
    src/reference_motion.cpp)
  target_include_directories(agi_x2_obs PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include)
  target_link_libraries(agi_x2_obs PUBLIC sonic_common)
  target_compile_features(agi_x2_obs PUBLIC cxx_std_20)

  if(AGI_X2_BUILD_ZMQ_HELPERS)
    add_library(agi_x2_zmq STATIC
      src/zmq_pose_input_source.cpp)
    target_include_directories(agi_x2_zmq PUBLIC
      ${CMAKE_CURRENT_LIST_DIR}/include
      ${ZMQ_INCLUDE_DIR}
      ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(agi_x2_zmq PUBLIC
      sonic_common
      ${ZMQ_LIBRARY})
    target_compile_features(agi_x2_zmq PUBLIC cxx_std_20)
  endif()

  enable_testing()
  add_executable(test_obs_builder test/test_obs_builder.cpp)
  target_link_libraries(test_obs_builder PRIVATE agi_x2_obs)
  add_test(NAME test_obs_builder COMMAND test_obs_builder)
  return()
endif()

# -----------------------------------------------------------------------------
# Full ROS 2 ament build path (the on-robot configuration).
# -----------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(aimdk_msgs REQUIRED)

# ZMQ (for VLA input + x2_debug PUB output) and nlohmann_json (for the
# packed-binary header). Both are header-only on the Python side; on
# Debian/Ubuntu we expect libzmq3-dev + nlohmann-json3-dev. On the X2
# dev box these are pre-installed by install_x2_deploy_deps.sh.
find_path(ZMQ_INCLUDE_DIR zmq.hpp
  PATHS /usr/include /usr/local/include)
find_library(ZMQ_LIBRARY zmq
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS /usr/include /usr/local/include)
if(NOT ZMQ_INCLUDE_DIR OR NOT ZMQ_LIBRARY)
  message(FATAL_ERROR
    "libzmq + cppzmq required for ZMQ VLA / debug helpers.\n"
    "Install via: apt install libzmq3-dev")
endif()
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
  message(FATAL_ERROR
    "nlohmann_json required for packed-binary header decoding.\n"
    "Install via: apt install nlohmann-json3-dev")
endif()
message(STATUS "ZMQ: ${ZMQ_LIBRARY} (include ${ZMQ_INCLUDE_DIR})")
message(STATUS "nlohmann_json: include ${NLOHMANN_JSON_INCLUDE_DIR}")

# ONNX Runtime is not a ROS package; locate it manually. ONNXRUNTIME_ROOT may
# point at either /opt/onnxruntime (prebuilt tarball layout) or a custom
# install prefix; we accept both.
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()
if(NOT ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_ROOT}/include
                ${ONNXRUNTIME_ROOT}/include/onnxruntime
                ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
             HINTS ${ONNXRUNTIME_ROOT}/lib)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime not found.\n"
    "  Searched ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}\n"
    "  Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime or install at /opt/onnxruntime.")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARY} (include ${ONNXRUNTIME_INCLUDE_DIR})")

# -----------------------------------------------------------------------------
# Library: obs builders + IO + ONNX actor + safety. Pure compile-time --
# no ROS 2 runtime startup is needed for the unit tests below.
# -----------------------------------------------------------------------------
add_library(agi_x2_deploy_lib STATIC
  src/proprioception_buffer.cpp
  src/tokenizer_obs.cpp
  src/reference_motion.cpp
  src/onnx_actor.cpp
  src/aimdk_io.cpp
  src/safety.cpp
  src/deploy_logger.cpp
  src/zmq_pose_input_source.cpp
  src/smpl_obs.cpp
  src/smpl_intent_source.cpp)
target_include_directories(agi_x2_deploy_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${ZMQ_INCLUDE_DIR}
  ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(agi_x2_deploy_lib PUBLIC
  sonic_common
  ${ONNXRUNTIME_LIBRARY}
  ${ZMQ_LIBRARY})
ament_target_dependencies(agi_x2_deploy_lib PUBLIC
  rclcpp sensor_msgs aimdk_msgs)

# -----------------------------------------------------------------------------
# Main executable.
# -----------------------------------------------------------------------------
add_executable(x2_deploy_onnx_ref src/x2_deploy_onnx_ref.cpp)
target_link_libraries(x2_deploy_onnx_ref PRIVATE agi_x2_deploy_lib)

# -----------------------------------------------------------------------------
# Tests.
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_obs_builder test/test_obs_builder.cpp)
  target_link_libraries(test_obs_builder agi_x2_deploy_lib)
endif()

# -----------------------------------------------------------------------------
# Install.
# -----------------------------------------------------------------------------
install(TARGETS x2_deploy_onnx_ref
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config/
        DESTINATION share/${PROJECT_NAME}/config)

ament_package()
