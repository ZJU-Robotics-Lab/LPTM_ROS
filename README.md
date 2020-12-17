# LPTM_ROS

## Installation Guide

* Add a CATKIN_IGNORE file in ```src/image_mcl/```
* Comment out ```catkin_python_setup()``` in ```src/lptm_ros/CMakeLists.txt```
* ```catkin_make```
* Delete CATKIN_IGNORE file in ```src/image_mcl/```
* Add a CATKIN_IGNORE file in ```src/lptm_ros/```
* Uncomment ```catkin_python_setup()``` in ```src/lptm_ros/CMakeLists.txt```
* ```catkin_make```


## Some Tips on tuning the MCL-DPCN on different dataset:

* Basic check:
  * local image size
  * global image size
  * rotation and ratio
  * the start point of the bag odom
  * the start point of the particle center
    * note that the two start point above could be different based on the start time of the rosbag
  * the orientation of the odometry
  * the validity of the local map (better be a complete photo and not the start of the bag)
* Advance:
  * DONNOT add the "255 - " in the weight choosing operation because it is not correct
  * check the max weight and the min weight given by DPCN, and tune the parametres to contrast them
  * make the the weight is in a reasonable range and that the wx+b does not make them nagative
