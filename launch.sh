#!/usr/bin/env bash
# =============================================================================
# launch.sh
# Portable launcher for the Kinova Gen3 Lite pick-and-place system.
#
# Features:
# - Discovers a usable ROS 2 setup automatically
# - Chooses a Python interpreter compatible with ROS build tooling
# - Rebuilds when the workspace is missing or stale
# - Accepts extra ros2 launch arguments from the command line
# - Supports debug helpers:
#     ROS_SETUP=/opt/ros/jazzy/setup.bash ./launch.sh
#     ROS_PYTHON=/usr/bin/python3 ./launch.sh
#     FORCE_BUILD=1 ./launch.sh
#     SKIP_BUILD=1 ./launch.sh
#     SKIP_LAUNCH=1 ./launch.sh
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SETUP="${SCRIPT_DIR}/install/setup.bash"

ROS_SETUP_OVERRIDE="${ROS_SETUP:-}"
ROS_PYTHON_OVERRIDE="${ROS_PYTHON:-}"
ROS_DISTRO_PREFERENCE="${ROS_DISTRO_OVERRIDE:-${ROS_DISTRO:-jazzy}}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"
LAUNCH_ARGS=("$@")

ROS_PYTHON_BIN=""

log() {
    printf '%s\n' "$1"
}

die() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

on_error() {
    local line_no="$1"
    printf '[ERROR] Command failed at line %s: %s\n' "${line_no}" "${BASH_COMMAND}" >&2
}

trap 'on_error "${LINENO}"' ERR

pick_ros_setup() {
    local candidate=""
    local fallback=""
    local distro=""
    local known_distros=(jazzy humble iron rolling)

    if [[ -n "${ROS_SETUP_OVERRIDE}" ]]; then
        [[ -f "${ROS_SETUP_OVERRIDE}" ]] || die "ROS_SETUP path not found: ${ROS_SETUP_OVERRIDE}"
        printf '%s\n' "${ROS_SETUP_OVERRIDE}"
        return
    fi

    if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
        printf '%s\n' "/opt/ros/${ROS_DISTRO}/setup.bash"
        return
    fi

    if [[ -f "/opt/ros/${ROS_DISTRO_PREFERENCE}/setup.bash" ]]; then
        printf '%s\n' "/opt/ros/${ROS_DISTRO_PREFERENCE}/setup.bash"
        return
    fi

    for distro in "${known_distros[@]}"; do
        if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
            candidate="/opt/ros/${distro}/setup.bash"
            break
        fi
    done

    if [[ -z "${candidate}" && -d /opt/ros ]]; then
        fallback="$(find /opt/ros -mindepth 2 -maxdepth 2 -path '*/setup.bash' | sort | head -n 1)"
        candidate="${fallback}"
    fi

    [[ -n "${candidate}" ]] || die "No ROS 2 setup found under /opt/ros. Set ROS_SETUP=/path/to/setup.bash."
    printf '%s\n' "${candidate}"
}

source_ros_environment() {
    local ros_setup
    local ros_name

    ros_setup="$(pick_ros_setup)"
    set +u
    # shellcheck disable=SC1090
    source "${ros_setup}"
    set -u
    ros_name="$(basename "$(dirname "${ros_setup}")")"
    log "[OK] Sourced ROS2 ${ros_name}"
}

pick_ros_python() {
    local candidate=""
    local resolved=""
    local seen=""
    local candidates=()

    if [[ -n "${ROS_PYTHON_OVERRIDE}" ]]; then
        candidates+=("${ROS_PYTHON_OVERRIDE}")
    fi
    candidates+=("/usr/bin/python3")
    have_cmd python3 && candidates+=("$(command -v python3)")
    have_cmd python && candidates+=("$(command -v python)")

    for candidate in "${candidates[@]}"; do
        [[ -n "${candidate}" ]] || continue
        [[ -x "${candidate}" ]] || continue
        resolved="$(readlink -f "${candidate}" 2>/dev/null || printf '%s' "${candidate}")"
        if [[ " ${seen} " == *" ${resolved} "* ]]; then
            continue
        fi
        seen="${seen} ${resolved}"
        if "${candidate}" -c "import catkin_pkg" >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    die "Could not find a Python interpreter with catkin_pkg. Install python3-catkin-pkg or set ROS_PYTHON=/path/to/python."
}

cleanup_process() {
    local process_name="$1"
    if have_cmd pkill; then
        pkill -9 -x "${process_name}" >/dev/null 2>&1 || true
    elif have_cmd killall; then
        killall -9 "${process_name}" >/dev/null 2>&1 || true
    fi
}

cleanup_leftovers() {
    log "[CLEANUP] Killing leftover Gazebo and ROS processes..."
    local processes=(
        gz
        ruby
        parameter_bridge
        robot_state_publisher
        spawner
        rqt_image_view
        color_detector_node
        pick_and_place_node
        ik_solver
    )
    local process_name

    for process_name in "${processes[@]}"; do
        cleanup_process "${process_name}"
    done

    if have_cmd ros2; then
        ros2 daemon stop >/dev/null 2>&1 || true
    fi

    sleep 2
    log "[OK] Cleanup complete"
}

require_command() {
    local command_name="$1"
    have_cmd "${command_name}" || die "Required command not found: ${command_name}"
}

workspace_needs_build() {
    if [[ "${SKIP_BUILD}" == "1" ]]; then
        return 1
    fi

    if [[ "${FORCE_BUILD}" == "1" ]]; then
        return 0
    fi

    if [[ ! -f "${INSTALL_SETUP}" ]]; then
        return 0
    fi

    if find "${SCRIPT_DIR}/src" -type f -newer "${INSTALL_SETUP}" -print -quit | grep -q .; then
        return 0
    fi

    return 1
}

build_workspace() {
    log "[BUILD] Building workspace..."
    (
        cd "${SCRIPT_DIR}"
        colcon build --symlink-install --cmake-force-configure --cmake-args \
            -DPython3_EXECUTABLE="${ROS_PYTHON_BIN}" \
            -DPYTHON_EXECUTABLE="${ROS_PYTHON_BIN}"
    )
    log "[OK] Build complete"
}

source_workspace() {
    [[ -f "${INSTALL_SETUP}" ]] || die "Workspace install overlay not found: ${INSTALL_SETUP}"
    set +u
    # shellcheck disable=SC1090
    source "${INSTALL_SETUP}"
    set -u
    log "[OK] Sourced workspace install"
}

log "=============================================="
log " Kinova Gen3 Lite Pick-and-Place Launcher"
log "=============================================="

[[ -d "${SCRIPT_DIR}/src" ]] || die "Workspace source directory not found: ${SCRIPT_DIR}/src"

source_ros_environment
require_command ros2
require_command colcon

cleanup_leftovers

ROS_PYTHON_BIN="$(pick_ros_python)"
export AMENT_PYTHON_EXECUTABLE="${ROS_PYTHON_BIN}"
export PYTHON_EXECUTABLE="${ROS_PYTHON_BIN}"
log "[OK] Using ROS Python: ${ROS_PYTHON_BIN}"

if workspace_needs_build; then
    build_workspace
else
    if [[ "${SKIP_BUILD}" == "1" ]]; then
        log "[WARN] SKIP_BUILD=1 set, reusing existing install overlay"
    else
        log "[OK] Workspace install is up to date"
    fi
fi

source_workspace

if [[ "${SKIP_LAUNCH}" == "1" ]]; then
    log "[OK] SKIP_LAUNCH=1 set, stopping after setup"
    exit 0
fi

log ""
log "Launching pick-and-place system..."
log "  ros2 launch kinova_description bringup.launch.py ${LAUNCH_ARGS[*]}"
log "=============================================="
log ""

exec ros2 launch kinova_description bringup.launch.py "${LAUNCH_ARGS[@]}"
