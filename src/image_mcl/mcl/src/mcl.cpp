#include "mcl.h"
#include "yaml.h"
YAML::Node config_params = YAML::LoadFile("/home/jessy104/ROS/LPTM_ROS/src/image_mcl/config/caoguangbiao.yaml");

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

  // gridMapCV = cv::imread("/home/mav-lab/Projects/Air-ground_matching/LPTM_ws/pixel_1_rot_black.png"); //grdiamp for use.

  gridMapCV = cv::imread(config_params["dir_to_global_map"].as<std::string>()); //grdiamp for use.


  // gridMapCV = tool::cvResizeMat(temp, 1 / 3.54);
  // gridMapCV = tool::cvRotateMat(temp, -27.8);
  cout<< "the map size is " << gridMapCV.cols << " " << gridMapCV.rows << endl;

  //--YOU CAN CHANGE THIS PARAMETERS BY YOURSELF--//
  numOfParticle = config_params["Particle_covariance"]["number_of_particle"].as<int>(); // Number of Particles.
  minOdomDistance = 0.01; // [m]
  minOdomAngle = 0; // [deg]
  repropagateCountNeeded = 1; // [num]
  odomCovariance[0] = 0.01; // Rotation to Rotation
  odomCovariance[1] = 0.01; // Translation to Rotation
  odomCovariance[2] = 0.01; // Translation to Translation
  odomCovariance[3] = 0.01; // Rotation to Translation

  odomCovariance[4] = config_params["Particle_covariance"]["X_Covariance"].as<float>(); // X
  odomCovariance[5] = config_params["Particle_covariance"]["Y_Covariance"].as<float>(); // Y
  template_size = config_params["Basic"]["local_image_size"].as<int>(); //gym 180// Template(square) size
  init_angle = config_params["Basic"]["rotation"].as<float>(); // Rotation init guess [degree]

  init_scale = config_params["Basic"]["scale_ratio"].as<float>();



  imageResolution = config_params["Basic"]["resolution"].as<float>()/init_scale; // [m] per [pixel]


  //--DO NOT TOUCH THIS PARAMETERS--//

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

  std::uniform_real_distribution<float> x_pos(0, template_size * imageResolution-2);
  std::uniform_real_distribution<float> y_pos(0, template_size * imageResolution-2); //heuristic setting. (to put particles into the map)


  //SET PARTICLES BY RANDOM DISTRIBUTION
  for(int i=0;i<numOfParticle;i++)
  {
    particle particle_temp;

    float randomX = x_pos(gen) + config_params["Odom_and_particles"]["start_location_particle_x"].as<float>()/init_scale;//gym27.5- 9;//26.5 - 9;//245.6/390*300; qsdjt
    float randomY = y_pos(gen) + config_params["Odom_and_particles"]["start_location_particle_y"].as<float>()/init_scale;//gym30.4- 9;//14.4 + 9;//245.6/390*300; qsdjt

    // float randomTheta = theta_pos(gen);
    particle_temp.pose = tool::xyzrpy2eigen(randomX,randomY,0,0,0,0);
    particle_temp.score = 1 / (double)numOfParticle;
    // particle_temp.theta = randomTheta / M_PI * 180;
    particles.push_back(particle_temp);
  }
  showInMap();
}

void mcl::prediction(Eigen::Matrix4f diffPose, cv::Mat local_measurement)
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
    particles.at(i).score +=1.e-30;
    particles.at(i).score = particles.at(i).score/scoreSum; // normalize the score
  }

}


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
  
  std::normal_distribution<double> dart(scoreBaseline, scoreBaseline/4);
  cout << "scoreBaseline" << scoreBaseline << endl;

  for(int i=0;i<particles.size();i++)
  {
    double darted = dart(gen); //darted number. (0 to maximum scores)
    // cout << "darted" << darted << endl;
    if(darted > scoreBaseline)
      darted = scoreBaseline - 1;
    auto lowerBound = std::lower_bound(particleScores.begin(), particleScores.end(), darted);
    int particleIndex = lowerBound - particleScores.begin(); // Index of particle in particles.
    // cout << "lowerBound" << lowerBound << endl;

    //TODO : put selected particle to array 'particleSampled' with score reset.

    particle selectedParticle = particles.at(particleIndex); // Which one you have to select?
    // selectedParticle.score = 1 / (double)particles.size();
    particleSampled.push_back(selectedParticle);

  }
  particles = particleSampled;
  // cout << "particle number: " << particles.size() << endl;
}


//DRAWING FUNCTION.
void mcl::showInMap(cv::Mat local_measurement)
{
//  cv::Mat showMap(gridMap.cols, gridMap.rows, CV_8UC3);
  cv::Mat showMap;
  showMap = gridMapCV.clone();
  // cv::cvtColor(gridMapCV, showMap, cv::COLOR_GRAY2BGR);
  int pred_x, pred_y;
  cout << "in show map with local_measurement" << endl;
  if(maxProbParticle.score >= 0)
  {
    //// Estimate position using all particles
    float x_all = 0;
    float y_all = 0;
    
    for(int i=0;i<particles.size();i++)
    {
      x_all = x_all + particles.at(i).pose(0,3) * particles.at(i).score;
      y_all = y_all + particles.at(i).pose(1,3) * particles.at(i).score;
    }
    pred_x = static_cast<int>(std::round((x_all) / imageResolution));
    pred_y = static_cast<int>(std::round((y_all) / imageResolution));

    local_measurement = tool::RotateImage(local_measurement, init_angle);
    local_measurement = tool::ResizeMat(local_measurement, init_scale);
   
    // cv::Mat alpha = tool::createAlpha(local_measurement);
    // tool::addAlpha(local_measurement, dst, alpha);
    cv::Mat merged;
    if(pred_x-local_measurement.cols/2 > 0 && pred_y-local_measurement.rows/2 > 0 && pred_x+local_measurement.cols/2 < showMap.cols && pred_y+local_measurement.rows/2 < showMap.rows){
      cv::Rect roi_rect = cv::Rect(pred_x-local_measurement.cols/2, pred_y-local_measurement.rows/2, local_measurement.cols, local_measurement.rows);

      cv::addWeighted( local_measurement, 0.5, showMap(roi_rect), 0.5, 0.0, merged);

      merged.copyTo(showMap(roi_rect));
      // local_measurement.copyTo(showMap(roi_rect));
      // cv::imshow("showmap", showMap(roi_rect));
      // cv::imshow("showmap", showMap(roi_rect));
      // cv::waitKey(1);
    }
  }
  for(int i=0;i<particles.size();i++)
  {
    float part_x = (particles.at(i).pose(0, 3)) / imageResolution;
    float part_y = (particles.at(i).pose(1, 3)) / imageResolution;

    int xPos  = static_cast<int>(std::round(part_x));
    int yPos = static_cast<int>(std::round(part_y));
    cv::circle(showMap,cv::Point(xPos,yPos),1,cv::Scalar(150,0,0),-1);
  }

  Eigen::VectorXf head_theta = tool::eigen2xyzrpy(Head_gt); // {x,y,z,roll,pitch,yaw}
  cout << "HEEEEEEEAAAAAAD"<< head_theta[5]/3.1415926*180.0<< endl;

  cv::circle(showMap,cv::Point(pose_show(0,3)/ imageResolution, pose_show(1,3)/ imageResolution),2,cv::Scalar(0,255,0),-1);
  cv::circle(showMap,cv::Point(pred_x,pred_y),2,cv::Scalar(0,0,255),-1);
  history_pred_xpos.push_back(pred_x);
  history_pred_ypos.push_back(pred_y);

  history_odom_xpos.push_back((odomFake(0,3)+cos(head_theta[5]))/imageResolution);
  history_odom_ypos.push_back((odomFake(1,3)-sin(head_theta[5]))/imageResolution);
  history_gt_xpos.push_back((pose_show(0,3)+cos(head_theta[5]))/ imageResolution);
  history_gt_ypos.push_back((pose_show(1,3)-sin(head_theta[5]))/ imageResolution);

  for(int i=0;i<history_pred_xpos.size();i++){
    cv::circle(showMap,cv::Point(history_pred_xpos[i],history_pred_ypos[i]),2,cv::Scalar(0,0,255),-1);
    cv::circle(showMap,cv::Point(history_odom_xpos[i],history_odom_ypos[i]),2,cv::Scalar(0,255,0),-1);
    cv::circle(showMap,cv::Point(history_gt_xpos[i],history_gt_ypos[i]),2,cv::Scalar(0,255,255),-1);
  }


  
  cv::imshow("MCL2", showMap);
  cv::waitKey(1);
}

void mcl::showInMap()
{
//  cv::Mat showMap(gridMap.cols, gridMap.rows, CV_8UC3);
  cv::Mat showMap;
  showMap = gridMapCV.clone();
  // cv::cvtColor(gridMapCV, showMap, cv::COLOR_GRAY2BGR);

  for(int i=0;i<particles.size();i++)
  {
    float part_x = (particles.at(i).pose(0, 3)) / imageResolution;
    float part_y = (particles.at(i).pose(1, 3)) / imageResolution;

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
    int xPos = static_cast<int>(std::round((x_all) / imageResolution));
    int yPos = static_cast<int>(std::round((y_all) / imageResolution));

    cv::circle(showMap,cv::Point(xPos,yPos),2,cv::Scalar(0,0,255),-1);
    // cv::Rect roi_rect = cv::Rect(xPos-template_size/2, yPos-template_size/2, local_measurement.cols, local_measurement.rows);
    // roi.copyTo(gridMapCV(roi_rect));

  }
  cv::circle(showMap,cv::Point(pose_show(0,3)/ imageResolution, pose_show(1,3)/ imageResolution),2,cv::Scalar(0,255,0),-1);

  cv::imshow("MCL2", showMap);
  cv::waitKey(1);
}

void mcl::LPTM(cv::Mat template_image, Eigen::Matrix4f current_pose)
{
  cv::Mat image;
  static int write_index = 0;
  image = tool::RotateImage(template_image, init_angle);
  image = tool::ResizeMat(image, init_scale);
  int particle_number = 0;
  int* particles_position_y, particles_position_x;
  image_mcl::image_coords img_coords;

  cv::Mat global_imroi;

  int rect_upperleft_x = int(current_pose(0,3) / imageResolution) - template_size/2;
  int rect_upperleft_y = int(current_pose(1,3) / imageResolution) - template_size/2;
  if(rect_upperleft_x < gridMapCV.cols - image.cols - 1 && rect_upperleft_y < gridMapCV.rows - image.rows - 1 && rect_upperleft_x > 0 && rect_upperleft_y > 0){
    cv::Rect rect(rect_upperleft_x, rect_upperleft_y, image.cols, image.rows);
    // cv::Rect rect(100, 100, 48, 48);
    cout << "rect_upperleft_x" << rect_upperleft_x << " rect_upperleft_y " << rect_upperleft_y << endl;
    global_imroi = gridMapCV(rect);
    
  }else if(rect_upperleft_x >= gridMapCV.cols - image.cols - 1){
    cv::Rect rect(gridMapCV.cols - image.cols-1, rect_upperleft_y, image.cols, image.rows);
    // cv::Rect rect(100, 100, 48, 48);
    global_imroi = gridMapCV(rect);
  }else if(rect_upperleft_y >= gridMapCV.rows - image.rows - 1){
    cv::Rect rect(rect_upperleft_x, gridMapCV.rows - image.rows-1, image.cols, image.rows);
    // cv::Rect rect(100, 100, 48, 48);
    global_imroi = gridMapCV(rect);
  }else if(rect_upperleft_y <= 0){
    cv::Rect rect(rect_upperleft_x, 0, image.cols, image.rows);
    // cv::Rect rect(100, 100, 48, 48);
    global_imroi = gridMapCV(rect);
  }else if(rect_upperleft_x <= 0){
    cv::Rect rect(0, rect_upperleft_y, image.cols, image.rows);
    // cv::Rect rect(100, 100, 48, 48);
    global_imroi = gridMapCV(rect);
  }

  cv::imshow("Aerial View", global_imroi);
  cv::waitKey(1);
  cv::imshow("Ground View", image);
  cv::waitKey(1);

  srv.request.x_position_of_particle.clear();
  srv.request.y_position_of_particle.clear();
  for(int i=0; i<particles.size(); i++)
  {
    srv.request.x_position_of_particle.push_back(int(particles.at(i).pose(0,3)/imageResolution) - rect_upperleft_x);
    srv.request.y_position_of_particle.push_back(int(particles.at(i).pose(1,3)/imageResolution) - rect_upperleft_y);
    
    // cout << "particle "<<int(particle_pose(0,3)/imageResolution) - rect_upperleft_x<<endl;
    // cout << "left pose " << rect_upperleft_y<< endl;
    particle_number ++;
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
      if(srv.response.weights_for_particle[i] > 1e-8)
      {
        particles.at(i).score = (srv.response.weights_for_particle[i] - 1.474) *1000;
      }else{
        particles.at(i).score = srv.response.weights_for_particle[i];
      }

      scoreSum += particles.at(i).score;
      // weights_visual[i] = particles.at(i).score;
      cout << "score" << particles.at(i).score << endl;
      if(maxScore < particles.at(i).score) // To check which particle has max score
      {
        maxProbParticle = particles.at(i);
        maxScore = particles.at(i).score;
      }
    }
    cout << "Max score" << maxScore << " " << maxProbParticle.pose(0, 3)/ imageResolution  - rect_upperleft_x << " " << maxProbParticle.pose(1, 3)/ imageResolution - rect_upperleft_y << endl;
    cout << "100 particle " << particles.at(100).pose(0, 3)/ imageResolution  - rect_upperleft_x << " " << particles.at(100).pose(1, 3)/ imageResolution  - rect_upperleft_y << endl;
    // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(srv.response.CorrMap, sensor_msgs::image_encodings::BGR8);
    // cv::Mat img = cv_ptr -> image;
    // cv::imshow("Corr Map", img);
    // cv::waitKey(1);
    for(int i=0;i<particles.size();i++)
    {
      particles.at(i).score +=1.e-300;
      particles.at(i).score = particles.at(i).score/scoreSum; // normalize the score
      maxScore = maxScore/scoreSum;
      // cout << "score" << particles.at(i).score << endl;  
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

void mcl::updateImageData(Eigen::Matrix4f pose, Eigen::Matrix4f Head_direction, cv::Mat local_measurement)
{
  cout<<"Name_of_Bag"<<config_params["Name_of_Bag"].as<string>()<<endl;
  if(!isOdomInitialized)
  {
    
    odomBefore = pose; // Odom used at last prediction.
    isOdomInitialized = true;
  }
  if (count_step <= 200){
    count_step++;
  }
  cout << "Count Step"<< count_step << endl;
  Head_gt = Head_direction;
  pose_show = pose;

  odomFake = pose;
  Eigen::Matrix4f odomFake_rot = tool::xyzrpy2eigen(0,0,0,0,0,0.009*count_step*3.14159/180.0);
  odomFake = odomFake_rot * odomFake;  
  std::uniform_real_distribution<float> odom_x_pos(0, 1);
  std::uniform_real_distribution<float> odom_y_pos(0, 1);
  // odomFake(0,3) += (odom_x_pos(gen) - 0)/1.155;//gym26.5;//27.5;//26.5;//220.5/390*300;  qsjdt; caogaungbiao 1.155
  // odomFake(1,3) += (odom_y_pos(gen) + 0.01*count_step)/(277.2-0.005*count_step)*120;
  // odomFake(0,3) += odom_x_pos(gen);//gym26.5;//27.5;//26.5;//220.5/390*300;  qsjdt; caogaungbiao 1.155
  // odomFake(1,3) += odom_y_pos(gen);

  Eigen::Matrix4f diffOdom =  pose * odomBefore.inverse(); // odom after = odom New * diffOdom
  Eigen::VectorXf diffxyzrpy = tool::eigen2xyzrpy(diffOdom); // {x,y,z,roll,pitch,yaw}
  float diffDistance = sqrt(pow(diffxyzrpy[0],2) + pow(diffxyzrpy[1],2));
  float diffAngle = fabs(diffxyzrpy[5]) * 180.0 / 3.141592;

  if(diffDistance>minOdomDistance || diffAngle>minOdomAngle)
  {
    //Doing correction & prediction
    cout << "Predicting step x = " << diffOdom(0,3) << " y = " << diffOdom(1,3) << endl;
    cout << "Pose step x = " << pose(0,3) << " y = " << pose(1,3) << endl;

    prediction(diffOdom, local_measurement);

    cout << "LPTM step" << endl;
    // cv::imshow("test", local_measurement);
    // cv::waitKey();
    // weightning_NCC(local_measurement);
  
    LPTM(local_measurement, odomFake);
    get_weights.Wait();
    showInMap(local_measurement);
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