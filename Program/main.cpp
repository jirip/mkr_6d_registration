#include <iostream>
#include <boost/graph/graph_concepts.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;
using namespace std;
using namespace pcl;

void print_match(DMatch match);
void print_keypoint(KeyPoint kpoint);
void nearest_keypoints_xy(vector<KeyPoint> & kpoints);

int main(int argc, char **argv) {
  
    cv::Mat scene_1, scene_2; // images to load
    
    scene_1 = imread("../data/desk_1_1.png", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    scene_2 = imread("../data/desk_1_2.png", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    
    //TODO: find the image features
    // parameters for SIFT detector
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    SiftFeatureDetector detector (nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    
    vector<KeyPoint> keypoints_1, keypoints_2; // sets of features Fa, Fb
    
    detector.detect( scene_1, keypoints_1);
    detector.detect( scene_2, keypoints_2);
    
    // Calculate descriptors (feature vectors)
    SiftDescriptorExtractor extractor;
    
    Mat descriptors_1, descriptors_2;
    
    extractor.compute( scene_1, keypoints_1, descriptors_1);
    extractor.compute( scene_2, keypoints_2, descriptors_2);
    
    // Matching descriptor vector using FLANN matcher
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match( descriptors_1, descriptors_2, matches);
    
    double max_dist = 0; double min_dist = 100; double scnd_min_dist = 100;
    
    // Calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors_1.rows; i++ )
    {
      double dist = matches[i].distance;
      //cout << dist << endl;
      if (dist < min_dist) {scnd_min_dist = min_dist; min_dist = dist;}
      if (dist > max_dist) max_dist = dist;
    }
    
    printf("-- Max dist : %f \n", max_dist );
    printf("-- 2nd min dist : %f \n", scnd_min_dist );
    printf("-- Min dist : %f \n", min_dist );
    
    // choose only good matches
    vector<DMatch> good_matches;
    
    for (int i = 0; i < descriptors_1.rows; i++)
    {
      if (matches[i].distance <= max(2*min_dist, 0.02))
      {
	good_matches.push_back( matches[i] );
      }
    }
    
    //TODO: Calculate Similarity
    double similarity = (double)good_matches.size()/(0.5*(keypoints_1.size()+keypoints_2.size()));
    cout << "Similarity: " << similarity << endl;
    
    // Draw only good matches
    Mat img_matches;
    drawMatches( scene_1, keypoints_1, scene_2, keypoints_2, good_matches,
		 img_matches, Scalar::all(-1), Scalar::all(-1),
		 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    imshow ("Good matches", img_matches);
    
    waitKey(0);
    
    // load depth data
    PointCloud<PointXYZRGB>::Ptr cloud1 (new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud2 (new PointCloud<PointXYZRGB>);
    io::loadPCDFile("../data/desk_1_1.pcd", *cloud1);
    io::loadPCDFile("../data/desk_1_2.pcd", *cloud2);
    
    //TODO: Estimating visual feature depth
    // Testing keypoints
    vector<KeyPoint> kp1,kp2;
    
    for (int i = 0; i < good_matches.size(); i++)
    {
      kp1.push_back(keypoints_1[good_matches[i].queryIdx]);
      kp2.push_back(keypoints_2[good_matches[i].trainIdx]);
    }
    
    /* round coordinates to nearest integer */
    nearest_keypoints_xy(kp1);
    nearest_keypoints_xy(kp2);
    
    /* Check if keypoints coordinates are rounded to nearest integer */
    /*for (int i = 0; i < good_matches.size(); i++)
    {
      print_keypoint(kp1[i]);
      print_keypoint(kp2[i]);
    }*/
    
    // create point cloud only for features
    PointCloud<PointXYZ>::Ptr depth_feat_1 (new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr depth_feat_2 (new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr depth_test (new PointCloud<PointXYZ>);
    
    Mat depth_1, depth_2;
    depth_1 = imread("../data/desk_1_1_depth.png", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
    depth_2 = imread("../data/desk_1_2_depth.png", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
    
    /* maybe not necessary */
    /*depth_1.convertTo(depth_1, CV_32F);
    depth_2.convertTo(depth_2, CV_32F);*/
    
    pcl::PointXYZ point;
    for (int i = 0; i < good_matches.size(); i++)
    {
      int x,y;
      x = kp1[i].pt.x;
      y = kp1[i].pt.y;
      uint16_t point_depth1 = (uint16_t)depth_1.at<uint16_t>(x,y);
      uint16_t point_depth2 = (uint16_t)depth_1.at<uint16_t>(x,y);
      if (point_depth1 || point_depth2){
	point.x = -(float)(x-240)/570.3/1000*point_depth1;
	point.y = (float)(y-320)/570.3/1000*point_depth1;
	point.z = (float)point_depth1/1000;
	depth_feat_1->points.push_back(point);
	x = kp2[i].pt.x;
	y = kp2[i].pt.y;
	point.x = -(float)(x-240)/570.3/1000*point_depth2;
	point.y = (float)(y-320)/570.3/1000*point_depth2;
	point.z = (float)point_depth2/1000;
	depth_feat_2->points.push_back(point);
      }
    }
    depth_feat_1->width = depth_feat_1->points.size();
    cout << "pocet 1: " << depth_feat_1->width << endl;
    depth_feat_1->height = 1;
    depth_feat_2->width = depth_feat_2->points.size();
    cout << "pocet 2: " << depth_feat_2->width << endl;
    depth_feat_2->height = 1;
    
    Eigen::Affine3f tr;
    getTransformation((float)1,(float)2,(float)3,(float)0.5,(float)1.5,(float)-2,tr);
    //cout << "Transformacni matice: " << tr << endl;
    transformPointCloud<PointXYZ>(*depth_feat_1, *depth_test, tr);
    
    /* Save keypoints clouds */
    io::savePCDFileBinary ("../data/kp1.pcd", *depth_feat_1);
    io::savePCDFileBinary ("../data/kp2.pcd", *depth_feat_2);
    
    /*visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(depth_feat_1);
    while(!viewer.wasStopped());*/
    
    /* ICP step */
    IterativeClosestPoint<PointXYZ, PointXYZ> icp;
    icp.setInputSource(depth_feat_1);
    icp.setInputTarget(depth_feat_2);
    
    cout << "Epsilon:" << (double)icp.getEuclideanFitnessEpsilon() << endl; 
    icp.setEuclideanFitnessEpsilon(icp.getEuclideanFitnessEpsilon()/10);
    
    PointCloud<PointXYZ>::Ptr registered (new PointCloud<PointXYZ>);
    
    icp.setMaximumIterations(10000);
    
    icp.align(*registered);
    cout << "Pocet final: " << registered->points.size() << endl;
    
    Eigen::Matrix4f transf = icp.getFinalTransformation();

    cout << transf << endl; 
    
    /* Transform original cloud */
    PointCloud<PointXYZRGB>::Ptr aligned (new PointCloud<PointXYZRGB>);
    transformPointCloud<PointXYZRGB>(*cloud1, *aligned, transf);
    io::savePCDFileBinary ("../data/output.pcd", *aligned);
    
    /*
    visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud1);
    viewer.showCloud(cloud2);
    while(!viewer.wasStopped(50));*/
    
    /*Mat img1, img2;
    drawKeypoints(scene_1, kp1, img1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(scene_2, kp2, img2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    imshow("Image1", img1);
    imshow("Image2", img2);*/
    
    cout << "ICP Fitness:" << (float)icp.getFitnessScore() << endl;
    cout << "Number of matches:" << (int)good_matches.size() << endl;
    cout << "Epsilon:" << icp.getEuclideanFitnessEpsilon() << endl; 
    return 0;
}

void print_match(DMatch match)
{
   cout << "Distance: " << match.distance << endl;
   cout << "Train image index: " << match.imgIdx << endl;
   cout << "Query descriptor index: " << match.queryIdx << endl;
   cout << "Train descriptor index: " << match.trainIdx << endl;
}

void print_keypoint(KeyPoint kpoint)
{
  cout << "X:" << kpoint.pt.x << endl;
  cout << "Y:" << kpoint.pt.y << endl;
}

void nearest_keypoints_xy(vector<KeyPoint> & kpoints)
{
  for (int i = 0; i < kpoints.size(); i++)
  {
    kpoints[i].pt.x = round(kpoints[i].pt.x);
    kpoints[i].pt.y = round(kpoints[i].pt.y);
  }
}