# LPTM_ROS

## Installation Guide

* Add a CATKIN_IGNORE file in ```src/image_mcl/```
* Comment out ```catkin_python_setup()``` in ```src/lptm_ros/CMakeLists.txt```
* ```catkin_make```
* Delete CATKIN_IGNORE file in ```src/image_mcl/```
* Add a CATKIN_IGNORE file in ```src/lptm_ros/```
* Uncomment ```catkin_python_setup()``` in ```src/lptm_ros/CMakeLists.txt```
* ```catkin_make```
