cmake_minimum_required(VERSION 2.8)
project(LoopDetector)
SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O0 -g  -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -g -march=native")

option(BUILD_DemoBRIEF  "Build demo application with BRIEF features" ON)
option(BUILD_DemoSURF   "Build demo application with SURF features"  ON)

set(HDRS
  include/DLoopDetector/DLoopDetector.h         include/DLoopDetector/TemplatedLoopDetector.h)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(catkin REQUIRED COMPONENTS roscpp rostime geometry_msgs tf std_msgs sensor_msgs message_filters cv_bridge image_transport compressed_image_transport compressed_depth_image_transport kinect2_bridge nav_msgs openslam_gmapping rosbag_storage)

find_package(OpenCV REQUIRED)
find_package(DLib REQUIRED)
find_package(DBoW2 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
MESSAGE(STATUS "include dir is" ${DBoW2_INCLUDE_DIRS})
MESSAGE(STATUS "lib dir is" ${DBoW2_LIBS})

include_directories(
include/DLoopDetector/ 
${OpenCV_INCLUDE_DIRS} 
${DLib_INCLUDE_DIRS} 
${DBoW2_INCLUDE_DIRS}
${catkin_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)

##############################################
## Declare ROS messages, services and actions ##
################################################

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES kinect2_viewer
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

# Display additional files in qtcreator
if(0)
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME}
  SOURCES
  ${FILES_LIST}
)
endif(0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

if(BUILD_DemoBRIEF)
  add_executable(demo_brief demo/demo_brief.cpp)
  target_link_libraries(demo_brief ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS} ${catkin_LIBRARIES})
endif(BUILD_DemoBRIEF)

if(BUILD_DemoSURF)
  add_executable(demo_surf demo/demo_surf.cpp)
  target_link_libraries(demo_surf ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS} ${catkin_LIBRARIES})
endif(BUILD_DemoSURF)

# My loop detector
add_executable(myloopdetector demo/MyLoopDetector.cpp)
target_link_libraries(myloopdetector ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS} ${catkin_LIBRARIES} ${EIGEN3_LIBS})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install(TARGETS myloopdetector demo_brief demo_surf
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

