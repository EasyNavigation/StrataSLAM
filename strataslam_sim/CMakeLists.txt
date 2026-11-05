cmake_minimum_required(VERSION 3.22)
project(strataslam_sim VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STRATASLAM_BUILD_TESTS "Build unit tests" ON)

if(NOT TARGET strataslam::core)
  find_package(strataslam_core REQUIRED)  # provides strataslam::core
endif()
find_package(OpenMP)

add_library(strataslam_sim SHARED src/building.cpp src/trajectory.cpp src/simulator.cpp src/scenarios.cpp)
target_include_directories(strataslam_sim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
add_library(strataslam::sim ALIAS strataslam_sim)
set_target_properties(strataslam_sim PROPERTIES EXPORT_NAME sim)
target_link_libraries(strataslam_sim PUBLIC strataslam::core)
if(OpenMP_CXX_FOUND)
  target_link_libraries(strataslam_sim PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(strataslam_sim PRIVATE -Wall -Wextra -Werror)
add_executable(strataslam_probe_lidar tools/probe_lidar.cpp)
target_link_libraries(strataslam_probe_lidar PRIVATE strataslam_sim)
add_executable(strataslam_probe_visual tools/probe_visual.cpp)
target_link_libraries(strataslam_probe_visual PRIVATE strataslam_sim)
add_executable(strataslam_probe_scenario tools/probe_scenario.cpp)
target_link_libraries(strataslam_probe_scenario PRIVATE strataslam_sim)
add_executable(strataslam_dump_scenario tools/dump_scenario.cpp)
target_link_libraries(strataslam_dump_scenario PRIVATE strataslam_sim)
add_executable(strataslam_fusion_ablation tools/fusion_ablation.cpp)
target_link_libraries(strataslam_fusion_ablation PRIVATE strataslam_sim)
add_executable(strataslam_fusion_ablation_synthetic tools/fusion_ablation_synthetic.cpp)
target_link_libraries(strataslam_fusion_ablation_synthetic PRIVATE strataslam::core)
target_compile_options(strataslam_fusion_ablation_synthetic PRIVATE -Wall -Wextra -Werror)
add_executable(strataslam_probe_pairs tools/probe_pairs.cpp)
target_link_libraries(strataslam_probe_pairs PRIVATE strataslam_sim)
add_executable(strataslam_scenario_runner tools/scenario_runner.cpp)
target_link_libraries(strataslam_scenario_runner PRIVATE strataslam_sim)
target_compile_options(strataslam_scenario_runner PRIVATE -Wall -Wextra -Werror)

if(STRATASLAM_BUILD_TESTS AND NOT TARGET GTest::gtest_main)
  find_package(GTest REQUIRED)  # standalone (colcon) build; the superbuild finds it once for all
endif()
if(STRATASLAM_BUILD_TESTS)
  include(CTest)
  include(GoogleTest)
  add_executable(strataslam_sim_tests test/building_test.cpp test/trajectory_test.cpp test/simulator_test.cpp)
  target_link_libraries(strataslam_sim_tests PRIVATE strataslam_sim GTest::gtest_main)
  # Listed when they run, not while building (see strataslam_core/CMakeLists.txt).
  gtest_discover_tests(strataslam_sim_tests DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST)
  # Scenario (integration) tests S01..S14: slower, labelled "integration".
  add_executable(strataslam_scenario_tests test/scenario_test.cpp)
  target_link_libraries(strataslam_scenario_tests PRIVATE strataslam_sim GTest::gtest_main)
  gtest_discover_tests(strataslam_scenario_tests DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS integration TIMEOUT 1800)
endif()

find_package(benchmark QUIET)
if(benchmark_FOUND)
  add_executable(strataslam_benchmarks bench/benchmarks.cpp)
  target_link_libraries(strataslam_benchmarks PRIVATE strataslam_sim benchmark::benchmark)
endif()

install(TARGETS strataslam_sim EXPORT strataslam_simTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT strataslam_simTargets NAMESPACE strataslam:: DESTINATION share/strataslam_sim/cmake)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/strataslam_simConfig.cmake
"include(CMakeFindDependencyMacro)
find_dependency(strataslam_core)
include(\"\${CMAKE_CURRENT_LIST_DIR}/strataslam_simTargets.cmake\")
")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/strataslam_simConfig.cmake DESTINATION share/strataslam_sim/cmake)

# Same warning policy as the library for every tool, test and benchmark (spec 12.9; audit AUD-019).
foreach(t strataslam_probe_lidar strataslam_probe_visual strataslam_probe_scenario strataslam_dump_scenario strataslam_fusion_ablation
          strataslam_probe_pairs strataslam_sim_tests strataslam_scenario_tests strataslam_benchmarks)
  if(TARGET ${t})
    target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
  endif()
endforeach()
