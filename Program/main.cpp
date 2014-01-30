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
#include <ctime>

/* Values for converting RGB and Depth values to XYZ PointCloud */
#define CAM_CALIB   570.3
#define DATA_SCALE  1000

/* Setting of choosing good mathces */
#define MIN_MULTIPLE  10

/* Point type for ICP */
#define ICP_POINT_TYPE  PointXYZ

using namespace cv;
using namespace std;
using namespace pcl;

void register_clouds(int scan1, int scan2);
void joint_pcds(int num1, int num2);
void print_match(DMatch match);
void print_keypoint(KeyPoint kpoint);
void nearest_keypoints_xy(vector<KeyPoint> & kpoints);

int main(int argc, char **argv) {
  
  /* Register 2 consecutive scans together in loop */
    for (int i = 1; i < 10; i++) {
      register_clouds(i,i+1);
    }
    
    cout << "Registration finished." << endl;
    
    return 0;
}

void register_clouds(int scan1, int scan2) {
  
    static int runned = 0;
    /* Load RGB images of scan 1 and scan 2*/
    Mat scene_1, scene_2; // images to load
    scene_1 = imread(format("../data/desk_1_%d.png",scan1), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    scene_2 = imread(format("../data/desk_1_%d.png",scan2), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    
    /* parameters for SIFT detector */
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    SiftFeatureDetector detector (nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    
    /* detect features in images (features Fa, Fb) */
    vector<KeyPoint> keypoints_1, keypoints_2; 
    detector.detect( scene_1, keypoints_1);
    detector.detect( scene_2, keypoints_2);
    
    /* Calculate descriptors (feature vectors) */
    SiftDescriptorExtractor extractor;
    Mat descriptors_1, descriptors_2;
    extractor.compute( scene_1, keypoints_1, descriptors_1);
    extractor.compute( scene_2, keypoints_2, descriptors_2);
    
    /* Matching descriptor vector using FLANN (Fast Library for Aproximate Nearest Neighbor) matcher */
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match( descriptors_1, descriptors_2, matches);
    
    double max_dist = 0; 
    double min_dist = 100; 
    double scnd_min_dist = 100;
    
    /* Find max and min distances (of descriptors) between keypoints */
    for (int i = 0; i < descriptors_1.rows; i++ )
    {
      double dist = matches[i].distance;
      if (dist < min_dist) {scnd_min_dist = min_dist; min_dist = dist;}
      if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist );
    printf("-- 2nd min dist : %f \n", scnd_min_dist );
    printf("-- Min dist : %f \n", min_dist );
    
    /* choose only "good" matches */
    vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
      /* Expression for good match */
      if (matches[i].distance <= max(MIN_MULTIPLE*min_dist, 0.02))
      {
	good_matches.push_back( matches[i] );
      }
    }
    
    cout << "Number of keypoints_1: " << keypoints_1.size() << endl;
    cout << "Number of keypoints_2: " << keypoints_2.size() << endl;
    cout << "Number of matches: " << matches.size() << endl;
    cout << "Number of GOOD matches: " << good_matches.size() << endl;
    
    /* Calculate Similarity */
    /* TODO Is that rigt? */
    float similarity = (float)good_matches.size()/(0.5*(keypoints_1.size()+keypoints_2.size()));
    cout << "Similarity: " << similarity << endl;
    
    /* Draw good matches (DEBUG)*/
    /*Mat img_matches, img_keypoints;
    drawMatches( scene_1, keypoints_1, scene_2, keypoints_2, good_matches,
		 img_matches, Scalar::all(-1), Scalar::all(-1),
		 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    namedWindow("Good matches", CV_WINDOW_AUTOSIZE | CV_GUI_EXPANDED);
    imshow ("Good matches", img_matches);*/
    
    /* Keypoints with rounded x y coordinates */
    vector<KeyPoint> kp1,kp2;
    for (int i = 0; i < good_matches.size(); i++)
    {
      kp1.push_back(keypoints_1[good_matches[i].queryIdx]);
      kp2.push_back(keypoints_2[good_matches[i].trainIdx]);
    }
    /* round coordinates to nearest integer */
    nearest_keypoints_xy(kp1);
    nearest_keypoints_xy(kp2);
    
    /* create point cloud only for matching features */
    PointCloud<PointXYZ>::Ptr depth_feat_1 (new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr depth_feat_2 (new PointCloud<PointXYZ>);
    //DEBUG PointCloud<PointXYZ>::Ptr depth_test (new PointCloud<PointXYZ>);
    
    /* Load depth data for scenes */
    Mat depth_1, depth_2;
    depth_1 = imread(format("../data/desk_1_%d_depth.png",scan1), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
    depth_2 = imread(format("../data/desk_1_%d_depth.png",scan2), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
    
    int rows_half = depth_1.rows/2;
    int cols_half = depth_1.cols/2;
    
    pcl::PointXYZ point1;
    pcl::PointXYZ point2;
    /*pcl::PointXYZ point; //DEBUG
    float d; // debugging
    for (int i = 0; i < depth_1.rows; i++)
    {
      for (int j = 0; j < depth_1.cols; j++)
      {
	d = (float)depth_1.at<uint16_t>(i,j);
	if (d) {
	  point.x = (float)(j-cols_half)/CAM_CALIB/DATA_SCALE*d;
	  point.y = (float)(i-rows_half)/CAM_CALIB/DATA_SCALE*d;
	  point.z = (float)d/DATA_SCALE;
	  depth_test->push_back(point);
	}
      }
    }
    io::savePCDFileASCII ("../data/depth1.pcd", *depth_test);*/
    /*
    visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(depth_test);
    while(!viewer.wasStopped());*/
    
    for (int i = 0; i < good_matches.size(); i++)
    {
      int x1,y1,x2,y2;
      /* Estimate depth for keypoints*/
      x1 = kp1[i].pt.x;
      y1 = kp1[i].pt.y;
      x2 = kp2[i].pt.x;
      y2 = kp2[i].pt.y;
      float point_depth1 = (float)depth_1.at<uint16_t>(y1,x1);     
      float point_depth2 = (float)depth_2.at<uint16_t>(y2,x2);
      /* //DEBUG
      if ((point_depth1 != 0) && (point_depth2 != 0)) {
	// Add point to depth_feat_1 cloud
	point1.x = -(float)(x1-rows_half)/CAM_CALIB/DATA_SCALE*point_depth1;
	point1.y = (float)(y1-cols_half)/CAM_CALIB/DATA_SCALE*point_depth1;
	point1.z = (float)point_depth1/DATA_SCALE;
	depth_feat_1->points.push_back(point1);
	// Add point to depth_feat_2 cloud
	point2.x = -(float)(x2-rows_half)/CAM_CALIB/DATA_SCALE*point_depth2;
	point2.y = (float)(y2-cols_half)/CAM_CALIB/DATA_SCALE*point_depth2;
	point2.z = (float)point_depth2/DATA_SCALE;
	depth_feat_2->points.push_back(point2);
      }*/
      
      if (point_depth1 != 0) {
	// Add point to depth_feat_1 cloud
	point1.x = (float)(x1-cols_half)/CAM_CALIB/DATA_SCALE*point_depth1;
	point1.y = (float)(y1-rows_half)/CAM_CALIB/DATA_SCALE*point_depth1;
	point1.z = (float)point_depth1/DATA_SCALE;
	depth_feat_1->points.push_back(point1);
	//cout << "x, y, depth: " << x1 << ", " << y1 << ", " << point_depth1 << endl;
	//cout << "Point1: " << point1 << endl;
      }
      if (point_depth2 != 0) {
	// Add point to depth_feat_2 cloud 
	point2.x = (float)(x2-cols_half)/CAM_CALIB/DATA_SCALE*point_depth2;
	point2.y = (float)(y2-rows_half)/CAM_CALIB/DATA_SCALE*point_depth2;
	point2.z = (float)point_depth2/DATA_SCALE;
	depth_feat_2->points.push_back(point2);
      }
    }
    
    depth_feat_1->width = depth_feat_1->points.size();
    depth_feat_1->height = 1;
    depth_feat_2->width = depth_feat_2->points.size();
    depth_feat_2->height = 1;
    
    cout << "pocet 1: " << depth_feat_1->width << endl;
    cout << "pocet 2: " << depth_feat_2->width << endl;
    
   
    /* Testovani funkce ICP*/
    /* //DEBUG
    Eigen::Affine3f tr;
    getTransformation((float)1,(float)2,(float)3,(float)0.5,(float)1.5,(float)-2,tr);
    //cout << "Transformacni matice: " << tr << endl;
    transformPointCloud<PointXYZ>(*depth_feat_1, *depth_test, tr);
    */
    
    /* Save keypoints clouds */
    io::savePCDFileASCII ("../data/kp1.pcd", *depth_feat_1);
    io::savePCDFileASCII ("../data/kp2.pcd", *depth_feat_2);
    
    //DEBUG
    /*visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(depth_feat_1);
    while(!viewer.wasStopped());*/
    
    PointCloud<PointXYZRGB>::Ptr cloud1 (new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud2 (new PointCloud<PointXYZRGB>);
    if(!runned)
      io::loadPCDFile(format("../data/desk_1_%d.pcd",scan1), *cloud1);
    else
      io::loadPCDFile("../data/output.pcd", *cloud1);
    io::loadPCDFile(format("../data/desk_1_%d.pcd",scan2), *cloud2);
    
    /* ICP step */
    IterativeClosestPoint<ICP_POINT_TYPE, ICP_POINT_TYPE> icp;
    icp.setInputSource(depth_feat_1);
    icp.setInputTarget(depth_feat_2);
    
    /* Set ICP parameters */
    icp.setMaximumIterations(1000);
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(0.00001);
    icp.setMaxCorrespondenceDistance(1000);
    
    PointCloud<ICP_POINT_TYPE>::Ptr aligned (new PointCloud<ICP_POINT_TYPE>);
    
    clock_t begin = clock();
    
    icp.align(*aligned);
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin)/double(CLOCKS_PER_SEC);
    cout << "Cas: " << elapsed_secs << endl;
    
    Eigen::Matrix4f transf = icp.getFinalTransformation();
    
    //DEBUG
    /* load full scene clouds data */
    //PointCloud<PointXYZRGB>::Ptr cloud1 (new PointCloud<PointXYZRGB>);
    //io::loadPCDFile(format("../data/desk_1_%d.pcd",123456789), *cloud1);
    
    /* Transform original cloud */
    PointCloud<PointXYZRGB>::Ptr transformed (new PointCloud<PointXYZRGB>);
    transformPointCloud<PointXYZRGB>(*cloud1, *transformed, transf);
    *transformed += *cloud2;
    io::savePCDFileBinary ("../data/output.pcd", *transformed);
    //io::savePCDFileBinary ("../data/reg.pcd", *aligned);
    
    /* //DEBUG
    visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud1);
    viewer.showCloud(cloud2);
    while(!viewer.wasStopped(50));*/
    
    /*Mat img1, img2;
    drawKeypoints(scene_1, kp1, img1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(scene_2, kp2, img2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    imshow("Image1", img1);
    imshow("Image2", img2);*/
    
    cout << transf << endl;
    cout << "ICP Fitness: " << icp.getFitnessScore() << endl;
    
    //DEBUG
    /*joint_pcds(1,scan2);/*
    joint_pcds(34,5);
    joint_pcds(56,7);
    joint_pcds(78,9);
    joint_pcds(9,10);*/
    runned++;
    //waitKey(0);
}

void joint_pcds(int num1, int num2)
{
  PointCloud<PointXYZRGB>::Ptr cloud1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr cloud2 (new PointCloud<PointXYZRGB>);
  io::loadPCDFile("../data/output.pcd", *cloud1);
  io::loadPCDFile(format("../data/desk_1_%d.pcd",num2), *cloud2);
  PointCloud<PointXYZRGB>::Ptr result (new PointCloud<PointXYZRGB>);
  *result = *cloud1 + *cloud2;
  io::savePCDFileASCII(format("../data/desk_1_%d%d.pcd",num1,num2), *result);
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