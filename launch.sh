#!/bin/bash
# =============================================================================
# launch.sh
# Convenience script to build and launch the Kinova Gen3 Lite pick-and-place
# system. Sources all required ROS2 workspaces and runs the bringup launch.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo " Kinova Gen3 Lite Pick-and-Place Launcher"
echo "=============================================="

# ---- Kill any leftover Gazebo / ROS processes from previous runs ----
echo "[CLEANUP] Killing leftover Gazebo and ROS processes..."
# Kill Gazebo server and client
killall -9 gz 2>/dev/null || true
killall -9 ruby 2>/dev/null || true
killall -9 parameter_bridge 2>/dev/null || true
killall -9 robot_state_publisher 2>/dev/null || true
killall -9 spawner 2>/dev/null || true
killall -9 rqt_image_view 2>/dev/null || true
killall -9 color_detector_node 2>/dev/null || true
killall -9 pick_and_place_node 2>/dev/null || true
killall -9 ik_solver 2>/dev/null || true

# Also kill any ros2 daemon
ros2 daemon stop 2>/dev/null || true

# Wait for processes to fully terminate
sleep 2
echo "[OK] Cleanup complete"

# Source ROS2 Jazzy
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    echo "[OK] Sourced ROS2 Jazzy"
else
    echo "[ERROR] ROS2 Jazzy not found at /opt/ros/jazzy/setup.bash"
    exit 1
fi

# Source the workspace (if built)
if [ -f "${SCRIPT_DIR}/install/setup.bash" ]; then
    source "${SCRIPT_DIR}/install/setup.bash"
    echo "[OK] Sourced workspace install"
else
    echo "[WARN] Workspace not yet built. Building now..."
    cd "${SCRIPT_DIR}"
    colcon build --symlink-install
    source "${SCRIPT_DIR}/install/setup.bash"
    echo "[OK] Build complete and sourced"
fi

echo ""
echo "Launching pick-and-place system..."
echo "  ros2 launch kinova_description bringup.launch.py"
echo "=============================================="
echo ""

ros2 launch kinova_description bringup.launch.py
