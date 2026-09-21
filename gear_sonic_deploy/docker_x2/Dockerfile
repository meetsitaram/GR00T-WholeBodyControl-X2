# X2 Ultra MuJoCo-sim container
#
# Sibling of gear_sonic_deploy/docker/Dockerfile.ros2 (which is the heavy
# CUDA + TensorRT + ONNX Runtime image used by the G1 deploy). This image is
# intentionally lighter -- ROS 2 Humble + aimdk_msgs + the Python deps the
# X2 MuJoCo<->ROS bridge needs.
#
# Modeled on AgiBot's lean record-and-replay reference Dockerfile (which is
# already battle-tested for talking to the real X2 Orin via DDS), with three
# changes for our use-case:
#   * builds aimdk_msgs from gear_sonic_deploy/thirdparty/aimdk_msgs/ (bring
#     your own: copy it from AgiBot's AimDK SDK -- see the README there --
#     or let install_scripts/setup_x2.sh --with-docker fetch it)
#   * drops the song-playback Python deps (lerobot, ruckig)
#   * adds the bridge's deps: mujoco, scipy, joblib, torch (CPU). torch is
#     CPU-only because eval_x2_mujoco.py imports it at module level for its
#     policy class -- the bridge itself never runs the policy, so we don't
#     need a GPU build.
#
# Build context: repo root (set by docker_x2/docker-compose.yml).
FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

# Apt deps:
#   python3-pip                          -- bridge's pip deps
#   ros-humble-sensor-msgs / geometry-msgs -- bridge pubs Imu, deploy eats them
#   ros-humble-rclcpp                    -- C++ deploy (build + link)
#   ros-humble-ament-cmake-gtest         -- C++ deploy unit tests
#   python3-colcon-common-extensions     -- `colcon build` the deploy
#   build-essential / cmake              -- C++ deploy compile chain
#   wget / ca-certificates               -- ONNX Runtime tarball download
#   curl / procps                        -- in-container debugging
#   libgl1 / libegl1 / libglu1-mesa      -- libGL.so.1 family (for OpenGL libs
#                                           when --sim-viewer is used; with
#                                           NVIDIA_DRIVER_CAPABILITIES=all the
#                                           NVIDIA driver libs override these
#                                           on the LD path -- mesa is the
#                                           graceful CPU fallback when there's
#                                           no NVIDIA runtime)
#   libglfw3                             -- GLFW window system used by
#                                           mujoco.viewer.launch_passive
#   libxcursor1 / libxinerama1 /         -- GLFW's X11 runtime deps; missing
#   libxrandr2 / libxi6 / libxext6 /        any of these and GLFW silently
#   libxrender1 / libxfixes3                fails to create a window
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-humble-sensor-msgs \
        ros-humble-geometry-msgs \
        ros-humble-rclcpp \
        ros-humble-ament-cmake-gtest \
        python3-colcon-common-extensions \
        build-essential \
        cmake \
        wget \
        ca-certificates \
        curl \
        procps \
        libgl1 \
        libegl1 \
        libglu1-mesa \
        libglfw3 \
        libxcursor1 \
        libxinerama1 \
        libxrandr2 \
        libxi6 \
        libxext6 \
        libxrender1 \
        libxfixes3 \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime 1.16.3 (x86_64 prebuilt tarball). Same version + install path
# `install_deps.sh` uses on the real X2 dev box, so the C++ deploy's CMake
# `find_path(... HINTS ${ONNXRUNTIME_ROOT}/include ...)` lands on the same
# layout in both worlds. Keeps `-DONNXRUNTIME_ROOT=/opt/onnxruntime` (the
# default `deploy_x2.sh` passes) hitting the right files.
ARG ONNXRUNTIME_VERSION=1.16.3
RUN cd /tmp && \
    wget -q --tries=3 \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
        -O onnxruntime.tgz && \
    tar xzf onnxruntime.tgz && \
    mv "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime && \
    rm onnxruntime.tgz && \
    echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf && \
    ldconfig
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

# Build aimdk_msgs from the SDK source staged under
# gear_sonic_deploy/thirdparty/aimdk_msgs/ (AgiBot's distribution; gitignored,
# fetched by install_scripts/setup_x2.sh --with-docker or copied by hand from
# the AimDK SDK's src/aimdk_msgs/). Upstream layout (interface/ + package.xml +
# CMakeLists.txt) is used verbatim. The build stops with a clear message when
# the directory only holds the README.
WORKDIR /ros2_ws/src
COPY gear_sonic_deploy/thirdparty/aimdk_msgs/ aimdk_msgs/
RUN test -f aimdk_msgs/package.xml || { \
        echo "ERROR: gear_sonic_deploy/thirdparty/aimdk_msgs/ has no package.xml." >&2; \
        echo "       Copy AgiBot's AimDK SDK src/aimdk_msgs/ there first (see the README in that dir)" >&2; \
        echo "       or run install_scripts/setup_x2.sh --with-docker." >&2; exit 1; }
WORKDIR /ros2_ws
RUN source /opt/ros/humble/setup.bash && \
    colcon build --packages-select aimdk_msgs \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# Python deps for the bridge.
#   mujoco           -- physics
#   numpy / scipy    -- math; scipy needed by eval_x2_mujoco for Rotation
#   joblib           -- motion-lib PKL loader (used when --sim-motion is set)
#   torch (CPU)      -- imported at module level by eval_x2_mujoco. The
#                       bridge does not run the policy; CPU build keeps the
#                       image ~1.5 GB lighter than the CUDA one.
#   pyzmq            -- Python bindings for libzmq3 (already installed
#                       below as a system dep). The bridge's OmniHand
#                       finger SUB needs this; without it, hand
#                       commands published by the recorder on tcp://5556
#                       are silently dropped and ``mj_data.ctrl`` for
#                       the fingers stays at the rest-pose seed --
#                       visible to the operator as "fingers don't move
#                       and just have noise on them" during teleop.
RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    pip3 install --no-cache-dir \
        mujoco \
        numpy \
        pyzmq \
        scipy \
        joblib && \
    pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

# Late-binding apt deps (kept in their own RUN layer so adding them does NOT
# invalidate the ONNX Runtime / aimdk_msgs / torch layers cached above --
# a full rebuild of those takes ~20 minutes).
#   iputils-ping        -- deploy_x2.sh's pre-flight reachability check
#   libzmq3-dev         -- C ZMQ lib (deploy harness ZmqPoseInputSource +
#                          x2_debug PUB; without it CMake aborts with a
#                          loud "libzmq + cppzmq required" message)
#   nlohmann-json3-dev  -- nlohmann/json.hpp for the packed-binary header
#                          decoder shared between the ZMQ source and the
#                          debug publisher
# cppzmq (zmq.hpp) is header-only and not packaged on Jammy as
# libcppzmq-dev, so we fetch the header from upstream and drop it into
# /usr/local/include where CMake's find_path resolves it.
ARG CPPZMQ_VERSION=4.10.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        iputils-ping \
        libzmq3-dev \
        nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/* && \
    wget -q --tries=3 \
        "https://raw.githubusercontent.com/zeromq/cppzmq/v${CPPZMQ_VERSION}/zmq.hpp" \
        -O /usr/local/include/zmq.hpp && \
    wget -q --tries=3 \
        "https://raw.githubusercontent.com/zeromq/cppzmq/v${CPPZMQ_VERSION}/zmq_addon.hpp" \
        -O /usr/local/include/zmq_addon.hpp

# Late-binding Python pip deps (kept separate from the heavy
# mujoco/scipy/torch layer so adding tools later doesn't invalidate the
# >5 minute torch CPU wheel + onnxruntime layers cached above).
#   matplotlib  -- offline analysis of rosbag data (joint trajectories,
#                  IMU traces, drift plots) for x2-nav-slam milestone 3+
RUN pip3 install --no-cache-dir matplotlib

# Default workdir is the bind-mounted monorepo (set in docker-compose.yml).
WORKDIR /workspace/sonic

# Drop the user into a shell with both ROS 2 and the aimdk_msgs overlay
# sourced. Override with `docker compose run x2sim <cmd>` for one-shots.
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec bash"]
