#ifndef MY_DETECTOR_HEAHER
#define MY_DETECTOR_HEAHER

#include <iostream>
#include <vector>
#include <string>
#include <boost/concept_check.hpp>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h>
#include "DLoopDetector.h"
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>
// Eigen
#include<Eigen/Dense>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

template <class TDescriptor>
class KeyFrame
{
public:
  KeyFrame(double &Timestamp, cv::Mat& image,cv::Mat&  depth): m_Timestamp(Timestamp),m_im(image), m_depth(depth) { }
 void setRt(const  tf::Transform & Rt)
 {
    m_Rt = Rt;
  }
  
  void setXY(const double &x, const double &y)
  {
    m_x = x;
    m_y = y;
  }
public:
  //size_t m_id;
  double m_Timestamp;
  cv::Mat m_im;
  cv::Mat m_depth;
   vector<cv::KeyPoint> m_keypoints;
  cv::Mat  m_descriptors;
  tf::Transform  m_Rt;
  //static  size_t  TotalNumber ;  //keyfame id
   
  double m_x;
  double m_y;
  
};


  
/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// @param TVocabulary vocabulary class (e.g: Surf64Vocabulary)
/// @param TDetector detector class (e.g: Surf64LoopDetector)
/// @param TDescriptor descriptor class (e.g: vector<float> for SURF)
template<class TVocabulary, class TDetector, class TDescriptor>
/// Class to run the demo 
class demoDetector
{
public:
  
  /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param width image width
   * @param height image height
   */
  demoDetector(const std::string &vocfile, int width, int height);
    
  ~demoDetector(){}

  /**
   * Runs the demo
   * @param name demo name
   * @param extractor functor to extract features
   */
  bool run(const std::string &name, 
    const FeatureExtractor<TDescriptor> &extractor, KeyFrame<TDescriptor> * newKF , unsigned int & match_Id);

  
protected:

  /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
  void readPoseFile(const char *filename, std::vector<double> &xs, 
    std::vector<double> &ys) const;
public:
  TVocabulary voc;
  TDetector detector;
    // Set loop detector parameters
  typename TDetector::Parameters params;
  
protected:

  std::string m_vocfile;
  int m_width;
  int m_height;
};

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
demoDetector<TVocabulary, TDetector, TDescriptor>::demoDetector
  (const std::string &vocfile, int width, int height)
  : m_vocfile(vocfile), m_width(width), m_height(height),  params(m_height, m_width)
{
    // We are going to change these values individually:
  params.use_nss = true; // use normalized similarity score instead of raw score
  params.alpha = 0.3; // nss threshold
  params.k = 1; // a loop must be consistent with 1 previous matches
  params.geom_check = GEOM_DI; // use direct index for geometrical checking
  params.di_levels = 2; // use two direct index levels
  
    // Load the vocabulary to use
  cout << "Loading   brief  vocabulary..." << endl;
  voc = TVocabulary (m_vocfile);
  detector = TDetector(voc, params);
  
   cout << " brief  vocabulary loaded." << endl;
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
bool demoDetector<TVocabulary, TDetector, TDescriptor>::run
  (const std::string &name, const FeatureExtractor<TDescriptor> &extractor, KeyFrame<TDescriptor> * newKF, unsigned int& match_Id)
{

  // Process images
  vector<cv::KeyPoint> keys;
  vector<TDescriptor> descriptors;
  
 // prepare profiler to measure times
  DUtils::Profiler profiler;
 
  // go
    cv::Mat im = newKF->m_im; // grey scale
    // get features
    profiler.profile("features");
    extractor(im, keys, descriptors);
    profiler.stop();
    
    // save keypoints and descriptors
    newKF->m_keypoints = keys;
    FBrief::toMat32F(descriptors, newKF->m_descriptors  );

    // add image to the collection and check if there is some loop
    DetectionResult result;
    
    profiler.profile("detection");
    detector.detectLoop(keys, descriptors, result);
    profiler.stop();
    
    if(result.detection())
    {
      cout << "- Loop found with image " << result.match << "!"
        << endl;
	//MyKeyFrame * matchkf = p_keyfames[result.match];
     // cout<<"the pose of "<<result.match<<" is "<<matchkf->m_x <<" "<<matchkf->m_y<<std::endl; 
      match_Id = result.match;
      return true;
    }
    else
    {
     return false;
    }
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::readPoseFile
  (const char *filename, std::vector<double> &xs, std::vector<double> &ys)
  const
{
  xs.clear();
  ys.clear();
  
  fstream f(filename, ios::in);
  
  string s;
  double ts, x, y, t;
  while(!f.eof())
  {
    getline(f, s);
    if(!f.eof() && !s.empty())
    {
      sscanf(s.c_str(), "%lf, %lf, %lf, %lf", &ts, &x, &y, &t);
      xs.push_back(x);
      ys.push_back(y);
    }
  }
  
  f.close();
}

// ---------------------------------------------------------------------------

#endif

