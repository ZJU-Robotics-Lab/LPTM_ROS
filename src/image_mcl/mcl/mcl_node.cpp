#include <ros/ros.h>
#include "src/mcl.h"
#include <message_filters/subscriber.h>
#include <boost/thread.hpp>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
using namespace std;

std::vector<Eigen::Matrix4f> vec_poses;
std::vector<double> vec_poses_time;
std::vector<Eigen::Matrix4Xf> vec_lasers;
std::vector<double>vec_lasers_time;
std::vector<cv::Mat> vec_images;
std::vector<double>vec_images_time;

// void callback(const nav_msgs::Odometry::ConstPtr & odom, const sensor_msgs::Image::ConstPtr & image);
typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::Image> MySyncPolicy;
class mcl_node
{
  public:
    mcl_node(ros::NodeHandle nh):nodeHandle_(nh), 
      mclocalizer(nh),
      subscribe_image(nh,"/elevation_mapping/orthomosaic", 10),
      subscribe_pose(nh, "/icp_odom", 10),
      sync(MySyncPolicy(10), subscribe_pose, subscribe_image)
    {
      sync.registerCallback(boost::bind(&mcl_node::callback, this, _1, _2));
    }

    ~mcl_node(){};

    void callback(const nav_msgs::Odometry::ConstPtr & odom, const sensor_msgs::Image::ConstPtr & image)
    {
      Eigen::Matrix4f eigenPose;
      tf::Quaternion q(odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z, odom->pose.pose.orientation.w);
      tf::Matrix3x3 m(q);
      // eigenPose<< m[0][0], m[0][1], m[0][2], odom->pose.pose.position.x,
      //             m[1][0], m[1][1], m[1][2], -odom->pose.pose.position.y,
      //             m[2][0], m[2][1], m[2][2], odom->pose.pose.position.z,
      //             0,0,0,1;
      cout << "hahahhahhahaha" << endl;
      eigenPose<< 1,0,0, odom->pose.pose.position.y,
                  0,1,0, odom->pose.pose.position.x,
                  0,0,1, odom->pose.pose.position.z,
                  0,0,0,1;
      // cout << "eigenPose" << eigenPose << endl;
      Eigen::Matrix4f static_rot = tool::xyzrpy2eigen(0,0,0,0,0,0);
      eigenPose = eigenPose * static_rot;
      eigenPose(0,3) += 174.2;
      eigenPose(1,3) += 19.8;
      // cout << "eigenPose x: " << odom->pose.pose.position.x << " y: " <<odom->pose.pose.position.y << endl;

      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
      cv::Mat img = cv_ptr -> image;

      mclocalizer.updateImageData(eigenPose, img);
    };

    mcl mclocalizer;

  private:
    message_filters::Subscriber<sensor_msgs::Image> subscribe_image;
    message_filters::Subscriber<nav_msgs::Odometry> subscribe_pose;
    ros::NodeHandle nodeHandle_;
    message_filters::Synchronizer<MySyncPolicy> sync;

};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "rs_mcl");
  ros::NodeHandle nh;
  
  mcl_node mcl_node_run(nh);
  ros::spin();

  return 0;
}
