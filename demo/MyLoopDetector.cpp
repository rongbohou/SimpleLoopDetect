
#include <iostream>
#include <vector>
#include <string>
// ROS
#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines BriefVocabulary
#include "DLoopDetector.h" // defines BriefLoopDetector
#include <DVision/DVision.h> // Brief

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "MyDetector.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

// ----------------------------------------------------------------------------

string VOC_FILE ;
string BRIEF_PATTERN_FILE ;
static const int IMAGE_W = 960; // image size
static const int IMAGE_H = 540;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<FBrief::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;

  /**
   * Creates the brief extractor with the given pattern file
   * @param pattern_file
   */
  BriefExtractor(const std::string &pattern_file);

private:

  /// BRIEF descriptor extractor
  DVision::BRIEF m_brief;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
typedef KeyFrame<FBrief::TDescriptor> KeyFrameType;

// PnP 结果
struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    int inliers;
};
// 相机内参结构
struct CAMERA_INTRINSIC_PARAMETERS 
{ 
    double cx, cy, fx, fy, scale;
};

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}

Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

         M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<float> toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

// cvMat2Eigen
tf::Transform cvMat2Rostf( cv::Mat& rvec, cv::Mat& tvec )
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
     vector<float> q = toQuaternion(R);
     tf::Transform T(tf::Quaternion(q[0], q[1], q[2],q[3]),tf::Vector3(tvec.at<float>(0),tvec.at<float>(1),tvec.at<float>(2)) );
     return T;
}

class ImageGrabber
{
public:
    ImageGrabber() :bfirst_listent(true),   demo(VOC_FILE,IMAGE_W, IMAGE_H),extractor(BRIEF_PATTERN_FILE)
    { 
      min_norm = 0.05;
      max_norm = 0.35;
      min_inliers = 5;
      // camera param
      camera.cx=496.180645820;
      camera.cy=282.322184731;
      camera.fx = 536.11987209565;
      camera.fy = 536.107766995;
      camera.scale=1000.0;
    }
    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);
    void PublishPose(cv::Mat Tcw);
    bool IsKeyFrame();
    RESULT_OF_PNP EstimateMotion( KeyFrameType& frame1, KeyFrameType& frame2, CAMERA_INTRINSIC_PARAMETERS& camera );
    
public:
    ros::Publisher* pPosPub;//发布位姿态
    tf::TransformBroadcaster* pbroadcaster;
    bool bfirst_listent;
    tf::StampedTransform T_basefoot_camera;
    tf::StampedTransform T_map_basefoot;
    tf::StampedTransform T_map_camera;
    tf::TransformListener mtf;// 订阅tf

    demoDetector<BriefVocabulary, BriefLoopDetector, FBrief::TDescriptor> demo;//(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);
    BriefExtractor extractor;//(BRIEF_PATTERN_FILE);
    std::vector<KeyFrame< FBrief::TDescriptor> *> p_keyfames;

    // keyframe threshold
    double min_norm ;
    double max_norm ;
    int min_inliers;
    CAMERA_INTRINSIC_PARAMETERS camera;

};

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat img = cv_ptrRGB->image;
    cv::Mat depth = cv_ptrD->image;
    double timestamp = cv_ptrRGB->header.stamp.toSec();
    
    // get the Rt
    // T_basefoot_camera
             if(bfirst_listent)
        {
          mtf.waitForTransform("base_footprint", "camera_rgb_optical_frame", ros::Time(), ros::Duration(1.0));
          mtf.lookupTransform("base_footprint", "camera_rgb_optical_frame",ros::Time(), T_basefoot_camera);
          std::cout<<" get the tf_basefoot_camera"<<std::endl;
          bfirst_listent =false;
        }
      mtf.lookupTransform("map", "base_footprint",ros::Time(), T_map_basefoot);
      mtf.lookupTransform("map", "camera_rgb_optical_frame",ros::Time(), T_map_camera);
   
      // is it a  new keyframe
     if(IsKeyFrame())
     {
       KeyFrameType * CurrentNewKF = new KeyFrameType(timestamp, img, depth);
       CurrentNewKF->setRt(T_map_camera);
       unsigned int match_Id;
        bool haveloop = demo.run("BRIEF", extractor, CurrentNewKF, match_Id);
	p_keyfames.push_back(CurrentNewKF);
	
	//if have a loop and need relocalization
	if(haveloop)
	{
	  KeyFrameType * LoopKF = p_keyfames[match_Id]; 
	  RESULT_OF_PNP result = EstimateMotion(*CurrentNewKF,*LoopKF, camera);
	  
	  cout<<"inliers: "<<result.inliers<<endl;
	  if ( result.inliers >  min_inliers ) //inliers不够，放弃该帧
	  {
	      tf::Transform T_loop_current = cvMat2Rostf(result.rvec, result.tvec);
	      tf::Transform T_map_current = LoopKF->m_Rt * T_loop_current;
	      tf::Transform T_map_base = T_map_current * T_basefoot_camera.inverse();
	     
	      // Broadcast tf 
	      pbroadcaster->sendTransform(tf::StampedTransform(T_map_base,
                                    ros::Time::now(),
                                    "map","base"));//T_world_base
	      // cout the result
	      tf::Vector3 v = T_map_base.getOrigin();
	      std::cout<<"t_map_base by loop detection: " <<v.getX()<<" "<<v.getY()<<" "<<v.getZ()<<std::endl;
	  }  
	}
       
    }   
}

bool ImageGrabber::IsKeyFrame()
{
  if(p_keyfames.size() == 0)
  return true;
  else
  {
     KeyFrameType * LastKF = p_keyfames[p_keyfames.size()-1];
     tf::Transform deltaT  = LastKF->m_Rt.inverse() * T_map_camera;
     tf::Vector3 v = deltaT.getOrigin();
     double v_norm = sqrt(v.dot(v));
     std::cout<<"the distance between current frame to last keyframe is  "<< v_norm<<" meters"<<std::endl;
     
     if(min_norm < v_norm &&  v_norm< max_norm)
       return true;
     else
       return false;
     
  }
}

// T_21
RESULT_OF_PNP ImageGrabber:: EstimateMotion( KeyFrameType& frame1, KeyFrameType& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    vector< cv::DMatch > matches;
    cv::FlannBasedMatcher matcher;
    RESULT_OF_PNP result;
    matcher.match( frame1.m_descriptors, frame2.m_descriptors, matches );
   
    cout<<"find total "<<matches.size()<<" matches."<<endl;
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    double good_match_threshold = 4.0;// can modify
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] );
    }

    cout<<"good matches: "<<goodMatches.size()<<endl;
    if (goodMatches.size() < 5)
    {
        result.inliers = -1;
        return result;
    }
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = frame1.m_keypoints[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = frame1.m_depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.m_keypoints[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }

   if (pts_obj.size() ==0 || pts_img.size()==0)
    {
        result.inliers = -1;
        return result;
    }
    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    cout<<"solving pnp"<<endl;
    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );

   
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;
}

int main(int argc, char **argv)
{
     ros::init(argc, argv, "MyLoopDetector");
    ros::start();
    
        if(argc != 3)
    {
        cerr << endl << "Usage: rosrun LoopDetector myloopdetector path_to_vocabulary paht_to_BRIEF_PATTERN_FILE" << endl;        
        ros::shutdown();
        return 1;
    }  
    VOC_FILE = string(argv[1]);
    BRIEF_PATTERN_FILE = string(argv[2]);
    
    tf::TransformBroadcaster broadcaster;
    ImageGrabber igb;
    igb.pbroadcaster = &(broadcaster);
    
    ros::NodeHandle nh;
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/kinect2/qhd/image_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/kinect2/qhd/image_depth_rect", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));
       
    while(ros::ok() )  
    {
       ros::spinOnce();//call back once
    }
    
  ros::shutdown();
  return 0;
}

// ----------------------------------------------------------------------------

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary
  
  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;
  
  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;
  
  m_brief.importPairs(x1, y1, x2, y2);
}

// ----------------------------------------------------------------------------

void BriefExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  // extract FAST keypoints with opencv
  const int fast_th = 20; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);
  
  // compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors);
}

// ----------------------------------------------------------------------------

