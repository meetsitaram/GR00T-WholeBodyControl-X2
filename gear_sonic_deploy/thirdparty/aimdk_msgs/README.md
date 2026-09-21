# aimdk_msgs (bring your own — from AgiBot's official SDK)

The deploy node and the docker sim image build against AgiBot's `aimdk_msgs`
ROS 2 interface package, which AgiBot distributes with the AimDK SDK:

    https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html

`setup_x2.sh --with-docker` fetches it automatically from AgiBot's
official SDK artifact (override the URL via `AIMDK_SDK_URL`). Manual
fallback: download the SDK and copy its `src/aimdk_msgs/` directory here,
so this directory contains `package.xml`, `CMakeLists.txt`, and the
`interface/` tree:

    cp -r <sdk>/src/aimdk_msgs/* gear_sonic_deploy/thirdparty/aimdk_msgs/

The docker build checks for it and stops with a clear message if missing.
The downloaded package is intentionally gitignored — it is AgiBot's
distribution, obtained from AgiBot.
Only needed for the deploy/docker paths — headless eval and the MuJoCo
viewers run without it.
