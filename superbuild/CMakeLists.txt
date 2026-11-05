# Development superbuild: builds the core and the simulator in one CMake tree, without ROS.
#   cmake -S superbuild -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build
# It lives in its own directory (with COLCON_IGNORE) so that colcon, run from a workspace, finds the
# five packages of the repository instead of treating the repository root as one CMake package.
cmake_minimum_required(VERSION 3.22)
project(strataslam LANGUAGES CXX)
enable_testing()
# Imported targets are directory-scoped: find them here so every subdirectory can use them.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(GTSAM REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTest REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../strataslam_core ${CMAKE_BINARY_DIR}/strataslam_core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../strataslam_sim ${CMAKE_BINARY_DIR}/strataslam_sim)
