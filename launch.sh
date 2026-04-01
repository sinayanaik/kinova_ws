#!/usr/bin/env bash
# =============================================================================
# launch.sh
# Portable launcher for the Kinova Gen3 Lite pick-and-place system.
#
# On first run it will:
#   1. Auto-detect or install a ROS 2 distribution (Jazzy / Humble)
#   2. Install all required apt packages (ros, gazebo, ros2_control, etc.)
#   3. Install all required Python pip packages (pinocchio, numpy, opencv, …)
#   4. Export all necessary environment variables
#   5. Build the workspace
#   6. Launch the system
#
# Override knobs (environment variables):
#     ROS_SETUP=/opt/ros/jazzy/setup.bash  — force a specific ROS setup
#     ROS_PYTHON=/usr/bin/python3           — force a specific Python
#     FORCE_BUILD=1                         — always rebuild
#     SKIP_BUILD=1                          — never rebuild
#     SKIP_LAUNCH=1                         — stop after setup
#     SKIP_DEPS=1                           — skip dependency installation
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SETUP="${SCRIPT_DIR}/install/setup.bash"
DEPS_STAMP="${SCRIPT_DIR}/.deps_installed"
VENV_DIR="${SCRIPT_DIR}/.venv"

ROS_SETUP_OVERRIDE="${ROS_SETUP:-}"
ROS_PYTHON_OVERRIDE="${ROS_PYTHON:-}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"
SKIP_DEPS="${SKIP_DEPS:-0}"
LAUNCH_ARGS=("$@")

ROS_PYTHON_BIN=""
DETECTED_DISTRO=""

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { printf '\033[1;32m[INFO]\033[0m  %s\n' "$1"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$1"; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2; exit 1; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

on_error() {
    printf '\033[1;31m[ERROR]\033[0m Command failed at line %s: %s\n' "$1" "${BASH_COMMAND}" >&2
}
trap 'on_error "${LINENO}"' ERR

# ── Detect / resolve ROS 2 distro ───────────────────────────────────────────
detect_ros_distro() {
    local distro=""
    local known_distros=(jazzy humble iron rolling)

    # Already sourced?
    if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
        DETECTED_DISTRO="${ROS_DISTRO}"
        return 0
    fi

    # Explicit override?
    if [[ -n "${ROS_SETUP_OVERRIDE}" && -f "${ROS_SETUP_OVERRIDE}" ]]; then
        DETECTED_DISTRO="$(basename "$(dirname "${ROS_SETUP_OVERRIDE}")")"
        return 0
    fi

    # Scan /opt/ros
    for distro in "${known_distros[@]}"; do
        if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
            DETECTED_DISTRO="${distro}"
            return 0
        fi
    done

    # Nothing found — try to install Jazzy
    DETECTED_DISTRO=""
    return 1
}

ros_setup_path() {
    if [[ -n "${ROS_SETUP_OVERRIDE}" && -f "${ROS_SETUP_OVERRIDE}" ]]; then
        printf '%s' "${ROS_SETUP_OVERRIDE}"
    else
        printf '%s' "/opt/ros/${DETECTED_DISTRO}/setup.bash"
    fi
}

# ── Dependency installation ──────────────────────────────────────────────────
install_ros2() {
    log "No ROS 2 installation found. Attempting to install ROS 2 Jazzy..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common curl >/dev/null
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq ros-jazzy-desktop >/dev/null
    DETECTED_DISTRO="jazzy"
    log "ROS 2 Jazzy installed"
}

install_apt_deps() {
    local distro="${DETECTED_DISTRO}"
    log "Installing apt dependencies for ROS 2 ${distro}..."

    local pkgs=(
        # Core build tools
        python3-colcon-common-extensions
        python3-rosdep
        python3-pip
        python3-vcstool

        # ROS 2 base + messages
        "ros-${distro}-rclpy"
        "ros-${distro}-sensor-msgs"
        "ros-${distro}-geometry-msgs"
        "ros-${distro}-std-msgs"
        "ros-${distro}-control-msgs"
        "ros-${distro}-trajectory-msgs"
        "ros-${distro}-tf2-ros"

        # ros2_control
        "ros-${distro}-ros2-control"
        "ros-${distro}-ros2-controllers"
        "ros-${distro}-controller-manager"
        "ros-${distro}-joint-state-broadcaster"
        "ros-${distro}-joint-trajectory-controller"
        "ros-${distro}-gripper-controllers"

        # Gazebo + bridge
        "ros-${distro}-ros-gz"
        "ros-${distro}-gz-ros2-control"

        # Robot description / URDF
        "ros-${distro}-robot-state-publisher"
        "ros-${distro}-xacro"
        "ros-${distro}-joint-state-publisher"
        "ros-${distro}-joint-state-publisher-gui"
        "ros-${distro}-launch-ros"

        # Perception
        "ros-${distro}-cv-bridge"
        "ros-${distro}-image-transport"
        "ros-${distro}-vision-opencv"
    )

    sudo apt-get update -qq
    # Install whatever is available; some meta-packages bundle others
    sudo apt-get install -y -qq "${pkgs[@]}" 2>/dev/null || {
        warn "Some apt packages were unavailable — installing individually..."
        for pkg in "${pkgs[@]}"; do
            sudo apt-get install -y -qq "${pkg}" 2>/dev/null || warn "  skipped: ${pkg}"
        done
    }
    log "apt dependencies installed"
}

# ── Virtual environment ──────────────────────────────────────────────────────
setup_venv() {
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        log "Virtual environment already exists: ${VENV_DIR}"
    else
        log "Creating virtual environment at ${VENV_DIR}..."
        python3 -m venv --system-site-packages "${VENV_DIR}"
        log "Virtual environment created"
    fi

    # Activate the venv
    set +u
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    set -u
    log "Activated venv ($(python3 --version), $(which python3))"

    # Upgrade pip inside venv
    python3 -m pip install --quiet --upgrade pip 2>/dev/null || true

    # Install all pip packages in one call so the resolver honours numpy<2
    log "Installing pip packages into venv..."
    python3 -m pip install --quiet "numpy<2" pin opencv-python pyyaml 2>/dev/null || {
        warn "Batch pip install had issues — installing individually..."
        python3 -m pip install --quiet "numpy<2" 2>/dev/null || warn "  numpy<2 failed"
        python3 -m pip install --quiet pin 2>/dev/null || warn "  pin failed"
        python3 -m pip install --quiet opencv-python 2>/dev/null || warn "  opencv-python failed"
        python3 -m pip install --quiet pyyaml 2>/dev/null || warn "  pyyaml failed"
    }

    # Safety net: force numpy<2 in case pin pulled in numpy>=2
    local np_ver
    np_ver="$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0")"
    if [[ "${np_ver%%.*}" -ge 2 ]]; then
        warn "numpy ${np_ver} detected — forcing downgrade to <2 for ROS compatibility"
        python3 -m pip install --quiet --force-reinstall "numpy<2" 2>/dev/null || true
    fi

    # Verify pinocchio is importable
    if ! python3 -c "import pinocchio" 2>/dev/null; then
        warn "pinocchio not importable via pip 'pin'; trying apt fallback..."
        sudo apt-get install -y -qq "ros-${DETECTED_DISTRO}-pinocchio" 2>/dev/null \
            || sudo apt-get install -y -qq python3-pinocchio 2>/dev/null \
            || warn "Could not install pinocchio — IK solver may fail"
    fi

    log "Venv pip packages installed"
}

init_rosdep() {
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
        log "Initializing rosdep..."
        sudo rosdep init 2>/dev/null || true
    fi
    rosdep update --rosdistro="${DETECTED_DISTRO}" 2>/dev/null || true
}

install_all_deps() {
    if [[ "${SKIP_DEPS}" == "1" && -f "${DEPS_STAMP}" ]]; then
        log "SKIP_DEPS=1 — skipping dependency installation"
        return
    fi

    if [[ -f "${DEPS_STAMP}" ]]; then
        log "Dependencies already installed (stamp: ${DEPS_STAMP})"
        return
    fi

    log "First-time setup: installing dependencies..."
    install_apt_deps
    init_rosdep

    # Resolve any remaining rosdep keys
    (
        cd "${SCRIPT_DIR}"
        rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true
    )

    date -Iseconds > "${DEPS_STAMP}"
    log "All dependencies installed"
}

# ── Source ROS + workspace ───────────────────────────────────────────────────
source_ros_environment() {
    local setup
    setup="$(ros_setup_path)"
    [[ -f "${setup}" ]] || die "ROS 2 setup not found: ${setup}"
    set +u
    # shellcheck disable=SC1090
    source "${setup}"
    set -u
    log "Sourced ROS 2 ${DETECTED_DISTRO} (${setup})"
}

pick_ros_python() {
    # If venv is active, prefer its python
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
        printf '%s' "${VIRTUAL_ENV}/bin/python3"
        return
    fi

    local candidates=()
    [[ -n "${ROS_PYTHON_OVERRIDE}" ]] && candidates+=("${ROS_PYTHON_OVERRIDE}")
    candidates+=("/usr/bin/python3")
    have_cmd python3 && candidates+=("$(command -v python3)")

    for c in "${candidates[@]}"; do
        [[ -x "${c}" ]] && { printf '%s' "${c}"; return; }
    done
    die "No usable Python 3 interpreter found"
}

# ── Environment variables ────────────────────────────────────────────────────
export_environment() {
    # ROS 2 domain (avoid cross-talk on shared networks)
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

    # ROS 2 middleware (default Fast-DDS)
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

    # Python
    export AMENT_PYTHON_EXECUTABLE="${ROS_PYTHON_BIN}"
    export PYTHON_EXECUTABLE="${ROS_PYTHON_BIN}"

    # Ensure nodes spawned by ros2 launch (which use /usr/bin/python3)
    # can find the venv packages (numpy<2) ahead of ~/.local packages.
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        local venv_sp="${VIRTUAL_ENV}/lib/python3.$(python3 -c 'import sys;print(sys.version_info.minor)')/site-packages"
        export PYTHONPATH="${venv_sp}:${PYTHONPATH:-}"
    fi

    # Gazebo — set resource paths so gz can find models/plugins
    local gz_resource="${SCRIPT_DIR}/install/kinova_description/share"
    export GZ_SIM_RESOURCE_PATH="${gz_resource}:${GZ_SIM_RESOURCE_PATH:-}"
    export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:-/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins}"

    # Locale (avoid ROS 2 UTF-8 warnings)
    export LANG="${LANG:-en_US.UTF-8}"
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"

    # Colcon defaults
    export COLCON_LOG_LEVEL="${COLCON_LOG_LEVEL:-warning}"

    log "Environment variables exported (DOMAIN=${ROS_DOMAIN_ID}, RMW=${RMW_IMPLEMENTATION})"
}

# ── Process cleanup ──────────────────────────────────────────────────────────
cleanup_process() {
    local name="$1"
    if have_cmd pkill; then
        pkill -9 -x "${name}" 2>/dev/null || true
    elif have_cmd killall; then
        killall -9 "${name}" 2>/dev/null || true
    fi
}

cleanup_leftovers() {
    log "Cleaning up leftover processes..."
    local procs=(gz ruby parameter_bridge robot_state_publisher spawner
                 rqt_image_view color_detector_node pick_and_place_node)
    for p in "${procs[@]}"; do cleanup_process "${p}"; done
    have_cmd ros2 && ros2 daemon stop >/dev/null 2>&1 || true
    sleep 2
    log "Cleanup done"
}

# ── Build ────────────────────────────────────────────────────────────────────
workspace_needs_build() {
    [[ "${SKIP_BUILD}" == "1" ]] && return 1
    [[ "${FORCE_BUILD}" == "1" ]] && return 0
    [[ ! -f "${INSTALL_SETUP}" ]] && return 0
    find "${SCRIPT_DIR}/src" -type f -newer "${INSTALL_SETUP}" -print -quit | grep -q . && return 0
    return 1
}

build_workspace() {
    log "Building workspace..."
    (
        cd "${SCRIPT_DIR}"
        colcon build --symlink-install --cmake-force-configure --cmake-args \
            -DPython3_EXECUTABLE="${ROS_PYTHON_BIN}" \
            -DPYTHON_EXECUTABLE="${ROS_PYTHON_BIN}"
    )
    log "Build complete"
}

source_workspace() {
    [[ -f "${INSTALL_SETUP}" ]] || die "Workspace install overlay not found: ${INSTALL_SETUP}"
    set +u
    # shellcheck disable=SC1090
    source "${INSTALL_SETUP}"
    set -u
    log "Sourced workspace overlay"
}

# ── Main ─────────────────────────────────────────────────────────────────────
log "=============================================="
log " Kinova Gen3 Lite Pick-and-Place Launcher"
log "=============================================="

[[ -d "${SCRIPT_DIR}/src" ]] || die "Source directory not found: ${SCRIPT_DIR}/src"

# Step 1: Detect or install ROS 2
if ! detect_ros_distro; then
    install_ros2
    detect_ros_distro || die "ROS 2 installation failed"
fi
log "Detected ROS 2 distro: ${DETECTED_DISTRO}"

# Step 2: Source ROS 2 underlay
source_ros_environment

# Step 3: Install dependencies (first run only)
install_all_deps

# Step 4: Create/activate venv and install pip packages
setup_venv

# Step 5: Pick Python and export env
ROS_PYTHON_BIN="$(pick_ros_python)"
export_environment
log "Using Python: ${ROS_PYTHON_BIN}"

# Step 6: Cleanup stale processes
cleanup_leftovers

# Step 7: Build if needed
if workspace_needs_build; then
    build_workspace
else
    if [[ "${SKIP_BUILD}" == "1" ]]; then
        warn "SKIP_BUILD=1 — reusing existing install"
    else
        log "Workspace up to date"
    fi
fi

# Step 8: Source workspace overlay
source_workspace

if [[ "${SKIP_LAUNCH}" == "1" ]]; then
    log "SKIP_LAUNCH=1 — stopping after setup"
    exit 0
fi

# Step 9: Launch
log ""
log "Launching pick-and-place system..."
log "  ros2 launch kinova_description bringup.launch.py ${LAUNCH_ARGS[*]:-}"
log "=============================================="
log ""

exec ros2 launch kinova_description bringup.launch.py "${LAUNCH_ARGS[@]}"
