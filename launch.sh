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
