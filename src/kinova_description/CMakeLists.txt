# =============================================================================
# kinova_description — URDF, meshes, worlds, configs, and launch files for
# the Kinova Gen3 Lite pick-and-place simulation.
# =============================================================================
cmake_minimum_required(VERSION 3.8)
project(kinova_description)

find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY urdf/ DESTINATION share/${PROJECT_NAME}/urdf)
install(DIRECTORY worlds/ DESTINATION share/${PROJECT_NAME}/worlds)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
