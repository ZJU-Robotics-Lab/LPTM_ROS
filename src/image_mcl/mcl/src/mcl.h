#ifndef MCL_H
#define MCL_H

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_mcl/image_coords.h>
#include <image_mcl/coords_weights.h>
#include <mutex>
#include <tf/tf.h>
#include <random>
#include "tool.h"
#include <cmath>
//#include <opencv2/core/cuda_devptrs.hpp>
//cuda headers
//#include "common.h"

#include <iostream>
#include <thread>
#include <condition_variable>
#include <lptm_ros/ComputePtWeights.h>

using namespace cv;

class mcl
{
  struct particle{
    Eigen::Matrix4f pose;
    float score;
    float theta;
    cv::Mat local_measurement;
    Eigen::Matrix4Xf scan; // Only for maximum probability particle.
  };

private:
  int m_sync_count;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen; //Standard mersenne_twister_engine seeded with rd()

  float imageResolution;
  float init_angle;
  float init_scale;
  float angle_search_area;
  float mapCenterX;
  float mapCenterY;
  float odomCovariance[6];
  int numOfParticle;
  std::vector<particle> particles;
  particle maxProbParticle;
  cv::Mat gridMap; // Gridmap for showing
  cv::Mat gridMapCV; // Gridmap for use (gaussian-blurred)
  Eigen::Matrix4f tf_laser2robot;
  Eigen::Matrix4f odomBefore;

  ros::NodeHandle nodeHandle_;
  ros::Publisher particle_pose_pub;
  ros::ServiceClient client;
  lptm_ros::ComputePtWeights srv;

  float minOdomDistance;
  float minOdomAngle;
  int template_size;
  int repropagateCountNeeded;
  bool receive_weights;
  bool first_flag;

  bool isOdomInitialized;
  int predictionCounter;

  std::mutex g_mutex;

  void initializeParticles();
  void prediction(Eigen::Matrix4f diffPose);
  void weightning(Eigen::Matrix4Xf laser);
  void weightning_NCC(cv::Mat template_image);
  void resampling();
  void LPTM(cv::Mat template_image, Eigen::Matrix4f pose, const nav_msgs::Odometry::ConstPtr & odom);
  void showInMap();

public:
  mcl(ros::NodeHandle nodeHandle);
  ~mcl();
  void updateLaserData(Eigen::Matrix4f pose, Eigen::Matrix4Xf laser);
  void updateImageData(Eigen::Matrix4f pose, cv::Mat local_measurement, const nav_msgs::Odometry::ConstPtr & odom);
  // float NCC(cv::Mat template_image, cv::Mat global_roi);
};


class Semaphore {
public:
	Semaphore(long count = 0)
		: count_(count) {
	}

    void Signal(unsigned int c = 1) {
		std::unique_lock<std::mutex> lock(mutex_);
        count_=c;
		cv_.notify_one();
	}

	void Wait() {
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(lock, [=] { return count_ > 0; });
		--count_;
	}

private:
	std::mutex mutex_;
	std::condition_variable cv_;
	long count_;
};


#endif