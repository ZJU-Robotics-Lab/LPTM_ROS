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
      subscribe_image(nh,"/elevation_mapping/orthomosaic", 2),
      subscribe_pose(nh, "/icp_odom", 2),
      sync(MySyncPolicy(5), subscribe_pose, subscribe_image)
    {
      sync.registerCallback(boost::bind(&mcl_node::callback, this, _1, _2));
      odom_pub_ = nodeHandle_.advertise<nav_msgs::Odometry>("trans_odom", 1);
    }

    ~mcl_node(){};

    void callback(const nav_msgs::Odometry::ConstPtr & odom, const sensor_msgs::Image::ConstPtr & image)
    {
      Eigen::Matrix4f eigenPose, Head_direction;
      tf::Quaternion q(odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z, odom->pose.pose.orientation.w);
      tf::Matrix3x3 m(q);
      Head_direction<< m[0][0], m[0][1], m[0][2], 0,
                  m[1][0], m[1][1], m[1][2], 0,
                  m[2][0], m[2][1], m[2][2], 0,
                  0,0,0,1;

      eigenPose<< 1,0,0, odom->pose.pose.position.y,
                  0,1,0, odom->pose.pose.position.x,
                  0,0,1, odom->pose.pose.position.z,
                  0,0,0,1;
      // cout << "eigenPose" << eigenPose << endl;

      Eigen::Matrix4f static_rot = tool::xyzrpy2eigen(0,0,0,0,0,-75.0*3.14159/180.0);
      Eigen::Matrix4f head_rot = tool::xyzrpy2eigen(0,0,0,0,0,-15.0*3.14159/180.0);
      eigenPose = static_rot * eigenPose;
      Head_direction = head_rot * Head_direction;
      eigenPose(0,3) += 218.0/380.0*300.0;//gym26.5;//27.5;//26.5;//220.5/390*300;  qsjdt
      eigenPose(1,3) += 8.0/380.0*300.0 ;//gym14.4;//25.5;//14.4;//9.2/390*300;  qsjdt

      // cout << "eigenPose x: " << odom->pose.pose.position.x << " y: " <<odom->pose.pose.position.y << endl;
      
      nav_msgs::Odometry odom_trans;
      odom_trans.header.frame_id = "/map";
      odom_trans.pose.pose.position.x = eigenPose(0,3);
      odom_trans.pose.pose.position.y = eigenPose(1,3);
      odom_trans.pose.pose.position.z = eigenPose(2,3);
      odom_trans.pose.pose.orientation.x = odom->pose.pose.orientation.x;
      odom_trans.pose.pose.orientation.y = odom->pose.pose.orientation.y;
      odom_trans.pose.pose.orientation.z = odom->pose.pose.orientation.z;
      odom_trans.pose.pose.orientation.w = odom->pose.pose.orientation.w;
      odom_pub_.publish(odom_trans);

      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
      cv::Mat img = cv_ptr -> image;

      mclocalizer.updateImageData(eigenPose, Head_direction, img);
    };

    mcl mclocalizer;

  private:
    message_filters::Subscriber<sensor_msgs::Image> subscribe_image;
    message_filters::Subscriber<nav_msgs::Odometry> subscribe_pose;
    ros::NodeHandle nodeHandle_;
    message_filters::Synchronizer<MySyncPolicy> sync;
    ros::Publisher odom_pub_;
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "rs_mcl");
  ros::NodeHandle nh;
  
  mcl_node mcl_node_run(nh);
  ros::spin();

  return 0;
}
