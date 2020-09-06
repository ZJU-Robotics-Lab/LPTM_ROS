cmake_minimum_required(VERSION 2.8.3)
project(lptm_ros)

find_package(catkin REQUIRED COMPONENTS roscpp rospy message_generation std_msgs geometry_msgs sensor_msgs)



add_service_files(
  DIRECTORY srv 
  FILES
  ComputePtWeights.srv
)

<<<<<<< HEAD
#catkin_python_setup()
=======
catkin_python_setup()
>>>>>>> b409f3f34681a07f1eb21ea261f04abb7e9c02d9

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  
  CATKIN_DEPENDS message_runtime   
)

#install(PROGRAMS detect
#        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
