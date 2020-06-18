#include "mcl.h"

using namespace std;
using namespace cv;
Semaphore get_weights(0);

mcl::mcl(ros::NodeHandle nodeHandle): nodeHandle_(nodeHandle)
{
  particle_pose_pub = nodeHandle.advertise<image_mcl::image_coords>("/particle_pose", 10);
  client = nodeHandle_.serviceClient<lptm_ros::ComputePtWeights>("compute_weight_server");
  
  m_sync_count = 0;
  first_flag = true;
  gen.seed(rd()); //Set random seed for random engine
  Mat temp;

  gridMapCV = cv::imread("/home/jessy104/ROS/LPTM_ws/src/pixel_1_rot.png"); //grdiamp for use.
  // gridMapCV = tool::cvResizeMat(temp, 1 / 3.54);
  // gridMapCV = tool::cvRotateMat(temp, -27.8);
  cout<< "the map size is " << gridMapCV.cols << " " << gridMapCV.rows << endl;

  //--YOU CAN CHANGE THIS PARAMETERS BY YOURSELF--//
  numOfParticle = 1000; // Number of Particles.
  minOdomDistance = 0.1; // [m]
  minOdomAngle = 0; // [deg]
  repropagateCountNeeded = 1; // [num]
  odomCovariance[0] = 0.01; // Rotation to Rotation
  odomCovariance[1] = 0.01; // Translation to Rotation
  odomCovariance[2] = 0.01; // Translation to Translation
  odomCovariance[3] = 0.01; // Rotation to Translation
  odomCovariance[4] = 0.11; // X
  odomCovariance[5] = 0.11; // Y
  template_size = 180; // Template(square) size
  init_angle = 0; // Rotation init guess [degree]
  init_scale = 2;
  angle_search_area = 3; // Searching area [degree]

  //--DO NOT TOUCH THIS PARAMETERS--//
  imageResolution = 0.1; // [m] per [pixel]
  tf_laser2robot << 1,0,0,0.0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1; // TF (laser frame to robot frame)
  mapCenterX = round(gridMapCV.cols/2) * imageResolution; // [m]
  mapCenterY = round(gridMapCV.rows/2) * imageResolution; // [m]
  isOdomInitialized = false; //Will be true when first data incoming.
  predictionCounter = 0;

  initializeParticles(); // Initialize particles.
  showInMap();
}

mcl::~mcl()
{

}

/* INITIALIZE PARTICLES UNIFORMLY TO THE MAP
 */
void mcl::initializeParticles()
{
  particles.clear();
  std::uniform_real_distribution<float> x_pos(0, template_size * imageResolution);
  std::uniform_real_distribution<float> y_pos(0, template_size * imageResolution); //heuristic setting. (to put particles into the map)

  //SET PARTICLES BY RANDOM DISTRIBUTION
  for(int i=0;i<numOfParticle;i++)
  {
    particle particle_temp;
    float randomX = x_pos(gen) +27.5;
    float randomY = y_pos(gen) +25.4;
    // float randomTheta = theta_pos(gen);
    particle_temp.pose = tool::xyzrpy2eigen(randomX,randomY,0,0,0,0);
    particle_temp.score = 1 / (double)numOfParticle;
    // particle_temp.theta = randomTheta / M_PI * 180;
    particles.push_back(particle_temp);
  }
  showInMap();
}

void mcl::prediction(Eigen::Matrix4f diffPose)
{
  std::cout<<"Predicting..."<<m_sync_count << " " << particles.size() <<std::endl;
  Eigen::VectorXf diff_xyzrpy = tool::eigen2xyzrpy(diffPose); // {x,y,z,roll,pitch,yaw} (z,roll,pitch assume to 0)

  //------------  FROM HERE   ------------------//
  //// Using odometry model
  double delta_trans = sqrt(pow(std::round(diff_xyzrpy(0)), 2)+ pow(std::round(diff_xyzrpy(1)),2));
  double delta_rot1 = atan2(diff_xyzrpy(1),diff_xyzrpy(0));
  double delta_rot2 = diff_xyzrpy(5) - delta_rot1;

  std::default_random_engine generator;
  if(delta_rot1  > M_PI)
          delta_rot1 -= (2*M_PI);
  if(delta_rot1  < -M_PI)
          delta_rot1 += (2*M_PI);
  if(delta_rot2  > M_PI)
          delta_rot2 -= (2*M_PI);
  if(delta_rot2  < -M_PI)
          delta_rot2 += (2*M_PI);
  //// Add noises to trans/rot1/rot2
  double trans_noise_coeff = odomCovariance[2]*fabs(delta_trans) + odomCovariance[3]*fabs(delta_rot1+delta_rot2);
  double rot1_noise_coeff = odomCovariance[0]*fabs(delta_rot1) + odomCovariance[1]*fabs(delta_trans);
  double rot2_noise_coeff = odomCovariance[0]*fabs(delta_rot2) + odomCovariance[1]*fabs(delta_trans);

  float scoreSum = 0;
  for(int i=0;i<particles.size();i++)
  {
    std::normal_distribution<double> gaussian_distribution(0, 1);

    delta_trans = delta_trans + gaussian_distribution(gen) * trans_noise_coeff;
    delta_rot1 = delta_rot1 + gaussian_distribution(gen) * rot1_noise_coeff;
    // delta_rot2 = delta_rot2 + gaussian_distribution(gen) * rot2_noise_coeff;

    // double x = delta_trans * cos(delta_rot1) + gaussian_distribution(gen) * odomCovariance[4];
    // double y = delta_trans * sin(delta_rot1) + gaussian_distribution(gen) * odomCovariance[5];

    double x = diff_xyzrpy(0) + gaussian_distribution(gen) * odomCovariance[4];
    double y = diff_xyzrpy(1) + gaussian_distribution(gen) * odomCovariance[5];
    // double theta = delta_rot1 + delta_rot2 + gaussian_distribution(gen) * odomCovariance[0]*(M_PI/180.0);

    Eigen::Matrix4f diff_odom_w_noise = tool::xyzrpy2eigen(x, y, 0, 0, 0, 0);

    Eigen::Matrix4f pose_t_plus_1 = particles.at(i).pose * diff_odom_w_noise;

    scoreSum = scoreSum + particles.at(i).score; // For normalization
    
    particles.at(i).pose = pose_t_plus_1;
    // particles.at(i).theta = theta / M_PI * 180;
  }

  //------------  TO HERE   ------------------//

  for(int i=0;i<particles.size();i++)
  {
    particles.at(i).score = particles.at(i).score/scoreSum; // normalize the score
  }
  showInMap();

}

// void mcl::weightning(Eigen::Matrix4Xf laser)
// {
//   float maxScore = 0;
//   float scoreSum = 0;

//   /* Your work.
//    * Input : laser measurement data
//    * To do : update particle's weight(score)
//    */

//   for(int i=0;i<particles.size();i++)
//   {

//     Eigen::Matrix4Xf transLaser = particles.at(i).pose* tf_laser2robot* laser; // now this is lidar sensor's frame.

//     //--------------------------------------------------------//

//     float calcedWeight = 0;

//     for(int j=0;j<transLaser.cols();j++)
//     {
//       int ptX  = static_cast<int>((transLaser(0, j) - mapCenterX + (300.0*imageResolution)/2)/imageResolution);
//       int ptY = static_cast<int>((transLaser(1, j) - mapCenterY + (300.0*imageResolution)/2)/imageResolution);

//       if(ptX<0 || ptX>=gridMapCV.cols || ptY<0 ||  ptY>=gridMapCV.rows) continue; // dismiss if the laser point is at the outside of the map.
//       else
//       {
//         double img_val =  gridMapCV.at<uchar>(ptY,ptX)/(double)255; //calculate the score.
//         calcedWeight += img_val; //sum up the score.
//       }


//     }
//     particles.at(i).score = particles.at(i).score + (calcedWeight / transLaser.cols()); //Adding score to particle.
//     scoreSum += particles.at(i).score;
//     if(maxScore < particles.at(i).score) // To check which particle has max score
//     {
//       maxProbParticle = particles.at(i);
//       maxProbParticle.scan = laser;
//       maxScore = particles.at(i).score;
//     }
//   }
//   for(int i=0;i<particles.size();i++)
//   {
//     particles.at(i).score = particles.at(i).score/scoreSum; // normalize the score
//   }
// }

void mcl::resampling()
{
  std::cout<<"Resampling..."<<m_sync_count<<std::endl;

  //Make score line (roullette)
  std::vector<double> particleScores;
  std::vector<particle> particleSampled;
  double scoreBaseline = 0;
  for(int i=0;i<particles.size();i++)
  {
    scoreBaseline += particles.at(i).score;
    particleScores.push_back(scoreBaseline);
  }

  std::uniform_real_distribution<double> dart(scoreBaseline/2, scoreBaseline);
  for(int i=0;i<particles.size();i++)
  {
    double darted = dart(gen); //darted number. (0 to maximum scores)
    auto lowerBound = std::lower_bound(particleScores.begin(), particleScores.end(), darted);
    int particleIndex = lowerBound - particleScores.begin(); // Index of particle in particles.

    //TODO : put selected particle to array 'particleSampled' with score reset.

    particle selectedParticle = particles.at(particleIndex); // Which one you have to select?
    selectedParticle.score = 1 / (double)particles.size();
    particleSampled.push_back(selectedParticle);

  }
  particles = particleSampled;
}

//DRAWING FUNCTION.
void mcl::showInMap()
{
//  cv::Mat showMap(gridMap.cols, gridMap.rows, CV_8UC3);
  cv::Mat showMap;
  showMap = gridMapCV.clone();
  // cv::cvtColor(gridMapCV, showMap, cv::COLOR_GRAY2BGR);

  for(int i=0;i<numOfParticle;i++)
  {
    float part_x = (particles.at(i).pose(0, 3)) / imageResolution - template_size/2;
    float part_y = (particles.at(i).pose(1, 3)) / imageResolution - template_size/2;

    int xPos  = static_cast<int>(std::round(part_x));
    int yPos = static_cast<int>(std::round(part_y));
    cv::circle(showMap,cv::Point(xPos,yPos),1,cv::Scalar(255,0,0),-1);
  }
  if(maxProbParticle.score > 0)
  {
    //// Original
  //  int xPos = static_cast<int>((maxProbParticle.pose(0, 3) - mapCenterX + (300.0*imageResolution)/2)/imageResolution);
  //  int yPos = static_cast<int>((maxProbParticle.pose(1, 3) - mapCenterY + (300.0*imageResolution)/2)/imageResolution);

    //// Estimate position using all particles
    float x_all = 0;
    float y_all = 0;
    for(int i=0;i<particles.size();i++)
    {
      x_all = x_all + particles.at(i).pose(0,3) * particles.at(i).score;
      y_all = y_all + particles.at(i).pose(1,3) * particles.at(i).score;
    }
    int xPos = static_cast<int>(std::round((x_all) / imageResolution) - template_size/2);
    int yPos = static_cast<int>(std::round((y_all) / imageResolution) - template_size/2);

    cv::circle(showMap,cv::Point(xPos,yPos),2,cv::Scalar(0,0,255),-1);

  }
  cv::circle(showMap,cv::Point(pose_show(0,3)/ imageResolution, pose_show(1,3)/ imageResolution),2,cv::Scalar(0,255,0),-1);

  cv::imshow("MCL2", showMap);
  cv::waitKey(1);
}

void mcl::LPTM(cv::Mat template_image, Eigen::Matrix4f current_pose)
{
  cv::Mat image;

  image = tool::cvRotateMat(template_image, init_angle);
  // image = tool::cvResizeMat(image,init_scale);
  int particle_number = 0;
  int* particles_position_y, particles_position_x;
  image_mcl::image_coords img_coords;

  cv::Mat global_imroi;
  int rect_upperleft_x = int(current_pose(0,3) / imageResolution) - template_size/2;
  int rect_upperleft_y = int(current_pose(1,3) / imageResolution) - template_size/2;
  cv::Rect rect(rect_upperleft_x, rect_upperleft_y, image.cols, image.rows);
  // cv::Rect rect(100, 100, 48, 48);
  global_imroi = gridMapCV(rect);
  
  cv::imshow("Aerial View", global_imroi);
  cv::waitKey(1);
  cv::imshow("Ground View", template_image);
  cv::waitKey(1);
  for(auto iter=particles.begin(); iter!=particles.end(); )
  {
    Eigen::Matrix4Xf particle_pose = iter->pose;
    float particle_rot = iter->theta;

    // if(iter->pose(0, 3) > 0 && iter->pose(1, 3) > 0 && iter->pose(0, 3) < (gridMapCV.cols - template_image.cols) * imageResolution 
    //   && iter->pose(1, 3) < (gridMapCV.rows - template_image.rows) * imageResolution){
    //   srv.request.x_position_of_particle.push_back((particle_pose(1,3)));
    //   srv.request.y_position_of_particle.push_back((particle_pose(0,3)));
    //   particle_number ++;
    // }else{
    //   iter->score = 0;
    // }
  
    srv.request.x_position_of_particle.push_back((particle_pose(0,3)/imageResolution) - rect_upperleft_x - template_size/2);
    srv.request.y_position_of_particle.push_back((particle_pose(1,3)/imageResolution) - rect_upperleft_y - template_size/2);
    // cout << "particle "<<particle_pose(0,3)/imageResolution<<endl;
    // cout << "left pose " << rect_upperleft_y<< endl;
    particle_number ++;
    iter++;
  }

  cv_bridge::CvImage cvimgt, cvimgs;
  sensor_msgs::Image t_msg, s_msg;
  cvimgt = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, image);
  cvimgt.toImageMsg(t_msg);
  cvimgs = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, global_imroi);
  cvimgs.toImageMsg(s_msg);

  srv.request.TemplateImage = t_msg;
  srv.request.SourceImage = s_msg;
  srv.request.particle_number = particle_number;

  if(client.call(srv))
  {
    ROS_INFO("particle number: %d, response: %d ", particles.size(), srv.response.weights_for_particle.size());

    maxScore = 0;
    float scoreSum = 0;

    for(auto i=0; i < srv.response.weights_for_particle.size(); i++)
    {
      particles.at(i).score = srv.response.weights_for_particle[i] * 100; 
      scoreSum += particles.at(i).score;
      weights_visual[i] = particles.at(i).score;
      // cout << "score" << particles.at(i).score << endl;
      if(maxScore < particles.at(i).score) // To check which particle has max score
      {
        maxProbParticle = particles.at(i);
        maxScore = particles.at(i).score;
      }
    }
    cout << "Max score" << maxScore << endl;
    for(int i=0;i<particles.size();i++)
    {
      particles.at(i).score = particles.at(i).score/scoreSum; // normalize the score
    }
    // first_flag = false;
    get_weights.Signal();
  }else
  {
    ROS_ERROR("Failed to call service");
  }
  
  // img_coords.header = odom->header;
  // img_coords.TemplateImage = t_msg;
  // img_coords.SourceImage = s_msg;
  // img_coords.particle_number = particle_number;
  // cout << odom->header << "header" << endl;
  // particle_pose_pub.publish(img_coords);
}

void mcl::updateImageData(Eigen::Matrix4f pose, cv::Mat local_measurement)
{
  if(!isOdomInitialized)
  {
    
    odomBefore = pose; // Odom used at last prediction.
    isOdomInitialized = true;
  }
  pose_show = pose;
  Eigen::Matrix4f diffOdom = odomBefore.inverse() * pose; // odom after = odom New * diffOdom
  Eigen::VectorXf diffxyzrpy = tool::eigen2xyzrpy(diffOdom); // {x,y,z,roll,pitch,yaw}
  float diffDistance = sqrt(pow(diffxyzrpy[0],2) + pow(diffxyzrpy[1],2));
  float diffAngle = fabs(diffxyzrpy[5]) * 180.0 / 3.141592;

  if(diffDistance>minOdomDistance || diffAngle>minOdomAngle)
  {
    //Doing correction & prediction
    // cout << "Predicting step x = " << diffOdom(0,3) << " y = " << diffOdom(1,3) << endl;
    // cout << "Pose step x = " << pose(0,3) << " y = " << pose(1,3) << endl;

    prediction(diffOdom);

    cout << "LPTM step" << endl;
    // cv::imshow("test", local_measurement);
    // cv::waitKey();
    // weightning_NCC(local_measurement);
  
    LPTM(local_measurement, pose);
    get_weights.Wait();

    predictionCounter++;
    if(predictionCounter == repropagateCountNeeded)
    {
      cout << "Resampling step" << endl;
      resampling();
      predictionCounter = 0;
    }

    m_sync_count = m_sync_count + 1;
    odomBefore = pose;
  }
}