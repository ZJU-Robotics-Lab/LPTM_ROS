cmake_minimum_required(VERSION 2.8.3)
project(image_mcl)
set (CMAKE_CXX_STANDARD 11)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    geometry_msgs
    nav_msgs
    std_msgs
    sensor_msgs
    message_filters
    cv_bridge
    message_generation
)


add_message_files(
  FILES
  coords_weights.msg
  image_coords.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)


include_directories(
  tool
  ${catkin_INCLUDE_DIRS}
)
message(  ${catkin_INCLUDE_DIRS})
find_package(OpenCV)
find_package(yaml-cpp)

# set cuda
find_package(CUDA REQUIRED)
message(STATUS "${CUDA_VERSION}")
message(STATUS "${CUDA_INCLUDE_DIRS}")
message(STATUS "${CUDA_LIBRARIES}")
set(CUDA_NVCC_FLAGS -G;-g;-std=c++11) # nvcc flags
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/include/yaml-cpp)



find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

set(tool_source
    tool/tool.cpp
    tool/tool.h
)

# MCL
set(mcl_source
    mcl/src/mcl.cpp
    mcl/src/mcl.h
)

add_subdirectory(mcl/src)
set (EXTRA_LIBS ${EXTRA_LIBS} mcl_gpu)
#add_executable(image_mcl mcl/mcl_node.cpp ${mcl_source} ${tool_source})

add_executable(image_mcl mcl/mcl_node.cpp ${tool_source})
#CUDA_ADD_EXECUTABLE(mcl mcl/src/ mcl.cu)
target_link_libraries(image_mcl ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_INCLUDE_DIRS} ${EXTRA_LIBS} yaml-cpp)





