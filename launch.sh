#!/usr/bin/env bash
# =============================================================================
# launch.sh
# Portable launcher for the Kinova Gen3 Lite pick-and-place system.
#
# On first run it will:
#   1. Auto-detect or install a ROS 2 distribution (Jazzy / Humble)
#   2. Install all required apt packages (ros, gazebo, ros2_control, etc.)
#   3. Create a project-local Python venv with compatible packages
#   4. Export all necessary environment variables
#   5. Build the workspace
#   6. Launch the system
#
# Override knobs (environment variables):
#     ROS_SETUP=/opt/ros/jazzy/setup.bash  — force a specific ROS setup
#     FORCE_BUILD=1                         — always rebuild
#     SKIP_BUILD=1                          — never rebuild
#     SKIP_LAUNCH=1                         — stop after setup
#     SKIP_DEPS=1                           — skip dependency installation
#     HEADLESS=1                            — run Gazebo without GUI
# =============================================================================
set -Euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SETUP="${SCRIPT_DIR}/install/setup.bash"
DEPS_STAMP="${SCRIPT_DIR}/.deps_installed"
VENV_DIR="${SCRIPT_DIR}/.venv"

ROS_SETUP_OVERRIDE="${ROS_SETUP:-}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"
SKIP_DEPS="${SKIP_DEPS:-0}"
HEADLESS="${HEADLESS:-0}"
LAUNCH_ARGS=("${@}")

DETECTED_DISTRO=""
MULTIARCH=""

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { printf '\033[1;32m[INFO]\033[0m  %s\n' "$1"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$1"; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2; exit 1; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

on_error() {
    printf '\033[1;31m[ERROR]\033[0m Script failed at line %s (cmd: %s)\n' "$1" "${BASH_COMMAND}" >&2
}
trap 'on_error "${LINENO}"' ERR

# Detect multiarch triplet once (e.g. x86_64-linux-gnu, aarch64-linux-gnu)
detect_multiarch() {
    if have_cmd dpkg-architecture; then
        MULTIARCH="$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
    fi
    if [[ -z "${MULTIARCH}" ]]; then
        local arch
        arch="$(uname -m)"
        case "${arch}" in
            x86_64)  MULTIARCH="x86_64-linux-gnu" ;;
            aarch64) MULTIARCH="aarch64-linux-gnu" ;;
            armv7l)  MULTIARCH="arm-linux-gnueabihf" ;;
            *)       MULTIARCH="${arch}-linux-gnu" ;;
        esac
    fi
}

# ── Detect / resolve ROS 2 distro ───────────────────────────────────────────
detect_ros_distro() {
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
    for d in "${known_distros[@]}"; do
        if [[ -f "/opt/ros/${d}/setup.bash" ]]; then
            DETECTED_DISTRO="${d}"
            return 0
        fi
    done

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

# ── ROS 2 installation (only when nothing found) ────────────────────────────
install_ros2() {
    log "No ROS 2 installation found. Attempting to install ROS 2 Jazzy..."

    sudo apt-get update -qq || die "apt-get update failed — check your internet"
    sudo apt-get install -y -qq software-properties-common curl locales >/dev/null

    sudo locale-gen en_US.UTF-8 2>/dev/null || true

    if [[ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]]; then
        sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
            -o /usr/share/keyrings/ros-archive-keyring.gpg \
            || die "Failed to download ROS GPG key — check your internet"
    fi

    local codename
    codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}")"
    [[ -n "${codename}" ]] || die "Cannot determine Ubuntu codename for ROS repo"

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu ${codename} main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

    sudo apt-get update -qq
    sudo apt-get install -y ros-jazzy-desktop >/dev/null 2>&1 \
        || die "Failed to install ros-jazzy-desktop"

    DETECTED_DISTRO="jazzy"
    log "ROS 2 Jazzy installed"
}

# ── Dependency installation ──────────────────────────────────────────────────
install_apt_deps() {
    local distro="${DETECTED_DISTRO}"
    log "Installing apt dependencies for ROS 2 ${distro}..."

    local pkgs=(
        # System / build tools
        python3-colcon-common-extensions
        python3-rosdep
        python3-pip
        python3-vcstool
        python3-venv
        locales
        curl
        git

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
        "ros-${distro}-controller-manager-msgs"
        "ros-${distro}-hardware-interface"

        # Gazebo Harmonic + ROS bridge
        "ros-${distro}-ros-gz"
        "ros-${distro}-gz-ros2-control"
        "ros-${distro}-ros-gz-sim"
        "ros-${distro}-ros-gz-bridge"
        "ros-${distro}-ros-gz-image"

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

        # Pinocchio (IK solver — best installed via apt for ABI compat)
        "ros-${distro}-pinocchio"

        # RMW (explicitly install so RMW_IMPLEMENTATION always resolves)
        "ros-${distro}-rmw-fastrtps-cpp"

        # rqt (used in bringup.launch.py)
        "ros-${distro}-rqt-image-view"
    )

    sudo apt-get update -qq || die "apt-get update failed"

    # Try bulk install first (fast path)
    if sudo apt-get install -y -qq "${pkgs[@]}" 2>/dev/null; then
        log "apt dependencies installed"
        return
    fi

    # Fallback: install one-by-one, track failures
    warn "Bulk install failed — installing packages individually..."
    local failed=()
    for pkg in "${pkgs[@]}"; do
        if ! sudo apt-get install -y -qq "${pkg}" 2>/dev/null; then
            failed+=("${pkg}")
        fi
    done

    if [[ ${#failed[@]} -gt 0 ]]; then
        warn "The following packages could not be installed:"
        for f in "${failed[@]}"; do warn "  - ${f}"; done
        # Check if critical packages are missing
        for crit in python3-venv python3-pip "ros-${distro}-rclpy" "ros-${distro}-ros-gz"; do
            for f in "${failed[@]}"; do
                [[ "${f}" == "${crit}" ]] && die "Critical package missing: ${crit}"
            done
        done
    fi
    log "apt dependencies installed (some optional packages skipped)"
}

# ── pip break-system-packages (PEP 668 on Ubuntu 23.04+) ────────────────────
ensure_pip_break_system_packages() {
    local pip_conf_dir="${HOME}/.config/pip"
    local pip_conf="${pip_conf_dir}/pip.conf"
    if [[ ! -f "${pip_conf}" ]] || ! grep -q 'break-system-packages' "${pip_conf}" 2>/dev/null; then
        mkdir -p "${pip_conf_dir}"
        printf '[global]\nbreak-system-packages = true\n' >> "${pip_conf}"
        log "Set pip break-system-packages in ${pip_conf}"
    fi
}

# ── Virtual environment ──────────────────────────────────────────────────────
setup_venv() {
    # Ensure python3-venv is available
    if ! python3 -m venv --help >/dev/null 2>&1; then
        log "python3-venv not found, installing..."
        sudo apt-get install -y -qq python3-venv 2>/dev/null \
            || die "Cannot install python3-venv — please install it manually"
    fi

    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        log "Virtual environment exists: ${VENV_DIR}"
    else
        log "Creating virtual environment at ${VENV_DIR}..."
        python3 -m venv --system-site-packages "${VENV_DIR}" \
            || die "Failed to create venv at ${VENV_DIR}"
        log "Virtual environment created"
    fi

    # Activate the venv (activate script references unset vars, so disable nounset)
    set +u
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate" || die "Failed to activate venv"
    set -u
    log "Activated venv ($(python3 --version 2>&1), $(which python3))"
}

install_venv_deps() {
    # Quick check: if numpy<2 is already in the venv, skip reinstalling
    local np_ver
    np_ver="$(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '0')"
    if [[ "${np_ver%%.*}" -lt 2 && "${np_ver}" != "0" ]]; then
        if python3 -c "import pinocchio" 2>/dev/null; then
            log "Venv packages already satisfied (numpy=${np_ver}, pinocchio ✓)"
            return
        fi
    fi

    log "Installing pip packages into venv..."

    # Upgrade pip first
    python3 -m pip install --quiet --upgrade pip 2>/dev/null || true

    # Install numpy<2 (critical — ROS cv_bridge and pinocchio are compiled
    # against NumPy 1.x and will segfault with NumPy 2.x)
    python3 -m pip install --quiet "numpy<2" \
        || die "Failed to install numpy<2 — check your internet"

    # Install other Python deps (opencv-python, pyyaml)
    # Note: do NOT install 'pin' via pip — it pulls in numpy>=2 via cmeel-boost.
    # Pinocchio is installed via apt (ros-${DETECTED_DISTRO}-pinocchio) and is
    # visible through --system-site-packages.
    python3 -m pip install --quiet opencv-python 2>/dev/null || {
        warn "pip install opencv-python failed — will use system cv2"
    }
    python3 -m pip install --quiet pyyaml 2>/dev/null || true

    # Final numpy version check
    np_ver="$(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '?')"
    if [[ "${np_ver%%.*}" -ge 2 ]]; then
        warn "numpy ${np_ver} still >=2 — forcing reinstall"
        python3 -m pip install --quiet --force-reinstall "numpy<2" \
            || die "Cannot enforce numpy<2 — this will cause runtime crashes"
        np_ver="$(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '?')"
    fi
    log "numpy version: ${np_ver} ✓"

    # Verify pinocchio (from apt, via system-site-packages)
    if ! python3 -c "import pinocchio" 2>/dev/null; then
        # Try pip install pin as last resort, then force numpy<2 again
        warn "pinocchio not importable from apt — trying pip install pin..."
        python3 -m pip install --quiet pin 2>/dev/null || true
        python3 -m pip install --quiet --force-reinstall "numpy<2" 2>/dev/null || true
        if ! python3 -c "import pinocchio" 2>/dev/null; then
            die "Cannot install pinocchio — IK solver will not work"
        fi
    fi
    log "pinocchio ✓"

    # Verify cv_bridge (comes from apt, visible via --system-site-packages)
    if ! python3 -c "from cv_bridge import CvBridge" 2>/dev/null; then
        warn "cv_bridge not importable — install ros-${DETECTED_DISTRO}-cv-bridge"
    fi

    log "Venv packages installed"
}

# ── rosdep ───────────────────────────────────────────────────────────────────
init_rosdep() {
    if ! have_cmd rosdep; then
        warn "rosdep not found — skipping"
        return
    fi
    # rosdep init is idempotent-ish; the error on re-init is harmless
    sudo rosdep init 2>/dev/null || true
    rosdep update --rosdistro="${DETECTED_DISTRO}" 2>/dev/null || true
}

install_all_deps() {
    if [[ "${SKIP_DEPS}" == "1" ]]; then
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

    # Resolve any remaining rosdep keys from package.xml files
    (
        cd "${SCRIPT_DIR}"
        rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true
    )

    # Locale generation (avoid UTF-8 warnings from ROS 2)
    sudo locale-gen en_US.UTF-8 2>/dev/null || true

    # Only stamp AFTER everything succeeds
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

# ── Environment variables ────────────────────────────────────────────────────
export_environment() {
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

    # Only set RMW if the package is actually installed
    local rmw="${RMW_IMPLEMENTATION:-}"
    if [[ -z "${rmw}" ]]; then
        if python3 -c "import rmw_fastrtps_cpp" 2>/dev/null \
           || [[ -d "/opt/ros/${DETECTED_DISTRO}/lib/librmw_fastrtps_cpp" ]] \
           || dpkg -s "ros-${DETECTED_DISTRO}-rmw-fastrtps-cpp" >/dev/null 2>&1; then
            rmw="rmw_fastrtps_cpp"
        fi
    fi
    if [[ -n "${rmw}" ]]; then
        export RMW_IMPLEMENTATION="${rmw}"
    fi

    # Python executables (for colcon cmake)
    local py_bin
    py_bin="$(which python3)"
    export AMENT_PYTHON_EXECUTABLE="${py_bin}"
    export PYTHON_EXECUTABLE="${py_bin}"

    # Prepend venv site-packages to PYTHONPATH so that nodes spawned by
    # ros2 launch (which use #!/usr/bin/python3) find numpy<2 from the venv
    # BEFORE the user's ~/.local numpy>=2.
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        local py_minor
        py_minor="$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo '12')"
        local venv_sp="${VIRTUAL_ENV}/lib/python3.${py_minor}/site-packages"
        if [[ -d "${venv_sp}" ]]; then
            export PYTHONPATH="${venv_sp}${PYTHONPATH:+:${PYTHONPATH}}"
        fi
    fi

    # Gazebo — resource paths (use detected multiarch, not hardcoded x86_64)
    local gz_resource="${SCRIPT_DIR}/install/kinova_description/share"
    export GZ_SIM_RESOURCE_PATH="${gz_resource}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"

    # Find Gazebo plugin path dynamically
    local gz_plugin_candidates=(
        "/usr/lib/${MULTIARCH}/gz-sim-8/plugins"
        "/usr/lib/${MULTIARCH}/gz-sim-7/plugins"
        "/opt/ros/${DETECTED_DISTRO}/opt/gz_sim_vendor/lib/gz-sim-8/plugins"
        "/opt/ros/${DETECTED_DISTRO}/lib/gz-sim-8/plugins"
    )
    for candidate in "${gz_plugin_candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
            export GZ_SIM_SYSTEM_PLUGIN_PATH="${candidate}${GZ_SIM_SYSTEM_PLUGIN_PATH:+:${GZ_SIM_SYSTEM_PLUGIN_PATH}}"
            break
        fi
    done

    # Locale
    export LANG="${LANG:-en_US.UTF-8}"
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"

    # Colcon log level
    export COLCON_LOG_LEVEL="${COLCON_LOG_LEVEL:-warning}"

    log "Environment exported (DOMAIN=${ROS_DOMAIN_ID}, RMW=${RMW_IMPLEMENTATION:-<default>}, ARCH=${MULTIARCH})"
}

# ── Process cleanup ──────────────────────────────────────────────────────────
cleanup_leftovers() {
    log "Cleaning up leftover processes..."
    # Only kill processes whose command line contains this workspace path
    # (avoids killing unrelated gz/ruby/ros processes on shared machines)
    local ws_path="${SCRIPT_DIR}"
    if have_cmd pkill; then
        pkill -9 -f "${ws_path}.*pick_and_place_node" 2>/dev/null || true
        pkill -9 -f "${ws_path}.*color_detector_node" 2>/dev/null || true
        pkill -9 -f "${ws_path}.*robot_state_publisher" 2>/dev/null || true
        pkill -9 -f "${ws_path}.*parameter_bridge" 2>/dev/null || true
        pkill -9 -f "gz sim.*pick_and_place" 2>/dev/null || true
    fi
    have_cmd ros2 && ros2 daemon stop >/dev/null 2>&1 || true
    sleep 2
    log "Cleanup done"
}

# ── Build ────────────────────────────────────────────────────────────────────
workspace_needs_build() {
    [[ "${SKIP_BUILD}" == "1" ]] && return 1
    [[ "${FORCE_BUILD}" == "1" ]] && return 0
    [[ ! -f "${INSTALL_SETUP}" ]] && return 0
    # Rebuild if any source file is newer than the install overlay
    if find "${SCRIPT_DIR}/src" -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.xacro' \
        -o -name '*.sdf' -o -name '*.urdf' -o -name 'CMakeLists.txt' -o -name 'package.xml' \
        -o -name 'setup.py' -o -name 'setup.cfg' \) -newer "${INSTALL_SETUP}" -print -quit \
        | grep -q .; then
        return 0
    fi
    return 1
}

build_workspace() {
    local py_bin
    py_bin="$(which python3)"
    log "Building workspace (python: ${py_bin})..."
    (
        cd "${SCRIPT_DIR}"
        colcon build --symlink-install --cmake-force-configure --cmake-args \
            -DPython3_EXECUTABLE="${py_bin}" \
            -DPYTHON_EXECUTABLE="${py_bin}" \
        2>&1 | tee /dev/stderr | tail -1
    ) || die "colcon build failed — see errors above"
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
main() {
    log "=============================================="
    log " Kinova Gen3 Lite Pick-and-Place Launcher"
    log "=============================================="

    [[ -d "${SCRIPT_DIR}/src" ]] || die "Source directory not found: ${SCRIPT_DIR}/src"

    # Detect hardware architecture
    detect_multiarch
    log "Architecture: ${MULTIARCH}"

    # Step 1: Detect or install ROS 2
    if ! detect_ros_distro; then
        install_ros2
        detect_ros_distro || die "ROS 2 installation failed"
    fi
    log "Detected ROS 2 distro: ${DETECTED_DISTRO}"

    # Step 2: Source ROS 2 underlay
    source_ros_environment

    # Step 3: Install apt dependencies (first run only)
    install_all_deps

    # Step 4: Allow pip to install alongside system packages (PEP 668)
    ensure_pip_break_system_packages

    # Step 5: Create/activate venv
    setup_venv

    # Step 6: Install pip packages into venv
    install_venv_deps

    # Step 7: Export environment
    export_environment

    # Step 8: Cleanup stale processes
    cleanup_leftovers

    # Step 9: Build if needed
    if workspace_needs_build; then
        build_workspace
    else
        if [[ "${SKIP_BUILD}" == "1" ]]; then
            warn "SKIP_BUILD=1 — reusing existing install"
        else
            log "Workspace up to date"
        fi
    fi

    # Step 10: Source workspace overlay
    source_workspace

    if [[ "${SKIP_LAUNCH}" == "1" ]]; then
        log "SKIP_LAUNCH=1 — stopping after setup"
        return 0
    fi

    # Step 11: Launch
    log ""
    log "Launching pick-and-place system..."
    log "  ros2 launch kinova_description bringup.launch.py ${LAUNCH_ARGS[*]:-}"
    log "=============================================="
    log ""

    exec ros2 launch kinova_description bringup.launch.py "${LAUNCH_ARGS[@]}"
}

main
