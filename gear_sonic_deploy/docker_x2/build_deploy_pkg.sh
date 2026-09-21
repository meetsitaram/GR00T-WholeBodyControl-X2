#!/usr/bin/env bash
# One-shot in-container build of the deploy package. Run via:
#   ./enter_sim.sh -- bash /workspace/sonic/gear_sonic_deploy/docker_x2/build_deploy_pkg.sh
# (enter_sim.sh already sources /opt/ros/humble + /ros2_ws/install setups.)
set -euo pipefail
cd /workspace/sonic/gear_sonic_deploy
# --base-paths: the root CMakeLists ('g1_deploy') shadows the tree; restrict
# discovery to the X2 package (mirrors deploy_x2.sh build_local()).
colcon build --packages-select agi_x2_deploy_onnx_ref \
    --base-paths src/x2/agi_x2_deploy_onnx_ref \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/opt/onnxruntime
echo "BUILD_DEPLOY_PKG_DONE rc=$?"
