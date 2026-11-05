cmake_minimum_required(VERSION 3.22)
project(strataslam_core VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(STRATASLAM_BUILD_TESTS "Build unit tests" ON)
option(STRATASLAM_ENABLE_SANITIZERS "Build with ASan + UBSan" OFF)
option(STRATASLAM_ENABLE_COVERAGE "Build with gcov instrumentation" OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenMP)
# No imgcodecs: on Ubuntu it loads GDAL and poppler (GPL). Image files are read with libpng/libjpeg (ADR-0047).
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc features2d calib3d)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

# small_gicp (MIT), pinned to v1.0.0. Built static; only used privately by the core.
include(FetchContent)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(BUILD_HELPER ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_Declare(small_gicp
  GIT_REPOSITORY https://github.com/koide3/small_gicp.git
  GIT_TAG fd29d8cf94cf05ed7ad21c81c27b65963110adb5  # commit of tag v1.0.0 (not the tag object bd48fd7)
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(small_gicp)


add_library(strataslam_core SHARED
  src/config_loader.cpp
  src/factors4.cpp
  src/fusion.cpp
  src/registration4.cpp
  src/colorize.cpp
  src/barometer.cpp
  src/imu_mount.cpp
  src/imu_preintegration.cpp
  src/gravity.cpp
  src/lidar_frontend.cpp
  src/depth_utils.cpp
  src/feature_extractor.cpp
  src/visual_tracker.cpp
  src/visual_frontend.cpp
  src/bow.cpp
  src/scan_context.cpp
  src/voxel_grid.cpp
  src/floor_graph.cpp
  src/backend.cpp
  src/slam_system.cpp
  src/floor_manager.cpp
  src/map_io.cpp
  src/metrics.cpp
  src/payloads_io.cpp
  src/pose4.cpp
  src/point_cloud_io.cpp
  src/floor_stacking.cpp src/logging.cpp
  src/transition.cpp)
add_library(strataslam::core ALIAS strataslam_core)
set_target_properties(strataslam_core PROPERTIES EXPORT_NAME core)  # installed as strataslam::core, same as in-tree

target_include_directories(strataslam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(strataslam_core PUBLIC Eigen3::Eigen gtsam PRIVATE yaml-cpp small_gicp ${OpenCV_LIBS})
target_include_directories(strataslam_core SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
  target_link_libraries(strataslam_core PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(strataslam_core PRIVATE -Wall -Wextra -Werror)

if(STRATASLAM_ENABLE_SANITIZERS)
  set(_san -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  # Under ASan Eigen stops assuming glibc malloc is aligned and uses its own aligned allocator,
  # while the (non-instrumented) system libgtsam frees with plain free(): bad-free across the ABI.
  # Make Eigen keep the same assumption as libgtsam (valid for 16-byte alignment on glibc x86_64).
  target_compile_definitions(strataslam_core PUBLIC EIGEN_MALLOC_ALREADY_ALIGNED=1)
  # GCC 13 emits false-positive maybe-uninitialized/array-bounds on inlined Eigen under ASan.
  target_compile_options(strataslam_core PUBLIC ${_san} -Wno-error=maybe-uninitialized -Wno-error=array-bounds)
  target_link_options(strataslam_core PUBLIC ${_san})
endif()
if(STRATASLAM_ENABLE_COVERAGE)
  target_compile_options(strataslam_core PUBLIC --coverage -O0 -g)
  target_link_options(strataslam_core PUBLIC --coverage)
endif()

# Image files for the tools (TUM run, vocabulary training), apart from the core library.
add_library(strataslam_image_io STATIC src/image_io.cpp)
add_library(strataslam::image_io ALIAS strataslam_image_io)
set_target_properties(strataslam_image_io PROPERTIES EXPORT_NAME image_io POSITION_INDEPENDENT_CODE ON)
target_include_directories(strataslam_image_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# The header uses GrayImage, whose measurements header includes Eigen and GTSAM: the same GTSAM as the core's (without
# it the target compiled against whatever GTSAM was on the default include path, or failed where there was none:
# the ros:jazzy container, whose GTSAM is only under /opt/ros).
target_link_libraries(strataslam_image_io PUBLIC Eigen3::Eigen gtsam PNG::PNG JPEG::JPEG)
target_compile_options(strataslam_image_io PRIVATE -Wall -Wextra -Werror)

add_executable(strataslam_write_sample_clouds tools/write_sample_clouds.cpp)
target_link_libraries(strataslam_write_sample_clouds PRIVATE strataslam_core)

add_executable(strataslam_tum_run tools/tum_run.cpp)
target_link_libraries(strataslam_tum_run PRIVATE strataslam_core strataslam_image_io)
add_executable(strataslam_fuse tools/fuse_tool.cpp)
target_link_libraries(strataslam_fuse PRIVATE strataslam_core)
add_executable(strataslam_eval_tool tools/eval_tool.cpp)
target_link_libraries(strataslam_eval_tool PRIVATE strataslam_core)
add_executable(strataslam_kitti_run tools/kitti_run.cpp)
target_link_libraries(strataslam_kitti_run PRIVATE strataslam_core)

# Same warning policy as the library for every tool (spec 12.9; audit AUD-019 found 16 translation units without it).
foreach(t strataslam_write_sample_clouds strataslam_tum_run strataslam_fuse strataslam_eval_tool strataslam_kitti_run)
  target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
endforeach()

if(STRATASLAM_BUILD_TESTS)
  include(CTest)  # enable_testing() plus the DartConfiguration that `colcon test` (ctest) looks for
  find_package(GTest REQUIRED)
  add_executable(strataslam_core_tests test/config_test.cpp test/io_test.cpp test/pose4_test.cpp test/metrics_test.cpp test/lidar_test.cpp test/visual_test.cpp test/floor_manager_test.cpp test/h6_test.cpp test/colorize_test.cpp test/fusion_test.cpp test/backend_test.cpp test/stacking_test.cpp test/imu_mount_test.cpp test/image_io_test.cpp test/ablation_factors_test.cpp)
  target_link_libraries(strataslam_core_tests PRIVATE strataslam_core strataslam_image_io GTest::gtest_main)
  target_compile_options(strataslam_core_tests PRIVATE -Wall -Wextra -Werror)
  target_compile_definitions(strataslam_core_tests PRIVATE
    STRATASLAM_PROFILE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../strataslam_ros/profiles")
  include(GoogleTest)
  # Tests are listed when they run, not while building: the build must not run the binary, which would load
  # the installed (previous) library first whenever the workspace's setup is sourced (LD_LIBRARY_PATH).
  gtest_discover_tests(strataslam_core_tests DISCOVERY_MODE PRE_TEST)
endif()

include(GNUInstallDirs)
install(TARGETS strataslam_core strataslam_image_io EXPORT strataslam_coreTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION bin)
install(TARGETS strataslam_fuse strataslam_eval_tool strataslam_write_sample_clouds RUNTIME DESTINATION lib/strataslam_core)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT strataslam_coreTargets NAMESPACE strataslam:: DESTINATION share/strataslam_core/cmake)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/strataslam_coreConfig.cmake
"include(CMakeFindDependencyMacro)
find_dependency(Eigen3)
find_dependency(GTSAM)
find_dependency(PNG)
find_dependency(JPEG)
include(\"\${CMAKE_CURRENT_LIST_DIR}/strataslam_coreTargets.cmake\")
")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/strataslam_coreConfig.cmake DESTINATION share/strataslam_core/cmake)
