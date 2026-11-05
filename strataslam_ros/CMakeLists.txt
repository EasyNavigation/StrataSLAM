cmake_minimum_required(VERSION 3.22)
project(strataslam_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(strataslam_core REQUIRED)
# The simulator is optional: without it (colcon build --packages-ignore strataslam_sim) the node, the launch
# files and the vocabulary trainer build, but not the bag exporter nor the tests that run on its bags.
find_package(strataslam_sim QUIET)
find_package(strataslam_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Same warning policy as the core (spec 12.9); ROS headers are system headers, so this covers our code.
set(STRATASLAM_WARNINGS -Wall -Wextra -Werror)

# ---- node (composable) ----
add_library(strataslam_node SHARED src/strataslam_node.cpp)
target_link_libraries(strataslam_node PUBLIC strataslam::core)
ament_target_dependencies(strataslam_node PUBLIC rclcpp rclcpp_components sensor_msgs nav_msgs geometry_msgs tf2_ros tf2_msgs message_filters std_srvs strataslam_msgs rosbag2_cpp ament_index_cpp)
target_compile_options(strataslam_node PRIVATE ${STRATASLAM_WARNINGS})
rclcpp_components_register_node(strataslam_node PLUGIN "strataslam_ros::StrataslamNode" EXECUTABLE strataslam_node_exe)
install(TARGETS strataslam_node LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)

# ---- simulator bag exporter ----
if(strataslam_sim_FOUND)
  add_executable(strataslam_sim_export_bag src/sim_bag_exporter.cpp)
  target_link_libraries(strataslam_sim_export_bag PRIVATE strataslam::sim strataslam::core)
  ament_target_dependencies(strataslam_sim_export_bag PUBLIC rclcpp rosbag2_cpp sensor_msgs geometry_msgs nav_msgs tf2_msgs strataslam_msgs)
  target_compile_options(strataslam_sim_export_bag PRIVATE ${STRATASLAM_WARNINGS})
  install(TARGETS strataslam_sim_export_bag RUNTIME DESTINATION lib/${PROJECT_NAME})
else()
  message(STATUS "strataslam_sim not found: strataslam_sim_export_bag and the tests on simulated bags are not built")
endif()

# ---- vocabulary trainer ----
add_executable(strataslam_vocab_train src/vocab_train.cpp)
# Images through strataslam::image_io (libpng, libjpeg), not OpenCV's imgcodecs, which loads GPL code (ADR-0047).
target_link_libraries(strataslam_vocab_train PRIVATE strataslam::core strataslam::image_io)
target_compile_options(strataslam_vocab_train PRIVATE ${STRATASLAM_WARNINGS})

install(TARGETS strataslam_vocab_train RUNTIME DESTINATION lib/${PROJECT_NAME})

# ---- python package + pybind11 module ----
ament_python_install_package(${PROJECT_NAME})
# NO_EXTRAS: no link-time optimization (its lto-wrapper notes went to stderr on every build).
pybind11_add_module(_strataslam_py MODULE NO_EXTRAS src/py_module.cpp)
target_link_libraries(_strataslam_py PRIVATE strataslam::core)
target_compile_options(_strataslam_py PRIVATE ${STRATASLAM_WARNINGS})
install(TARGETS _strataslam_py DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})

install(DIRECTORY launch profiles vocabularies DESTINATION share/${PROJECT_NAME})
# Live view in Rerun (viewer:=true); Rerun itself is a pip package (see docs/user/mapping.md).
install(PROGRAMS scripts/strataslam_viewer DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  ament_add_pytest_test(test_profiles test/test_profiles.py TIMEOUT 120)
  ament_add_pytest_test(test_launch_validation test/test_launch_validation.py TIMEOUT 120)
  ament_add_pytest_test(test_viewer test/test_viewer.py TIMEOUT 120)
  ament_add_pytest_test(test_malformed_clouds test/test_malformed_clouds.py TIMEOUT 180)
  ament_add_pytest_test(test_launch_instances test/test_launch_instances.py TIMEOUT 400)
  ament_add_pytest_test(test_tf_tree test/test_tf_tree.py TIMEOUT 240)
  if(strataslam_sim_FOUND)
    ament_add_pytest_test(test_node_with_profiles test/test_node_with_profiles.py TIMEOUT 900)
    ament_add_pytest_test(test_doctor_bags test/test_doctor_bags.py TIMEOUT 900)
  endif()
endif()

ament_package()
