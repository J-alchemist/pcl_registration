#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include "../inc/tic_toc.h"
/*
 * NDT内部将目标点云体素网格化（可设置网格大小），计算每个网格分布，
 * -->通过调整 T目标_源， 将源点云转到目标点云坐标系，计算最佳分布，输出最佳的 T目标_源.
 */

//ndt算法
pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt; 
void setNdtParams();
//icp算法
pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
void setIcpParams();
// 滤波算法
void voxelInit(pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud);
void voxelInit2(pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud);
void movePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr& obj_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& out_cloud);
void myDisplay(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
                pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud2, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud2);

// main
int main (int argc, char** argv) { 
  /*******************************通用处理*************************************/

  //加载 目标点云：target_cloud   源点云：source_cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("../src/key_frame_0.pcd", *target_cloud) == -1)     //目标点云
  {
    PCL_ERROR ("Couldn't read file key_frame_0.pcd \n");
    return (-1);
  }
  std::cout << "Loaded target cloud size: " << target_cloud->size () << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  // if (pcl::io::loadPCDFile<pcl::PointXYZ> ("../src/key_frame_0.pcd", *source_cloud) == -1)    //源点云
  // {
  //   PCL_ERROR ("Couldn't read file room_scan2.pcd \n");
  //   return (-1);
  // }
  // std::cout << "Loaded source cloud size: " << source_cloud->size ()  << std::endl;
  movePoints(target_cloud, source_cloud);     // 将目标点云做一点变换得到源点云

  // 体素滤波器 
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud (new pcl::PointCloud<pcl::PointXYZ>);    
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source_cloud (new pcl::PointCloud<pcl::PointXYZ>);    
  voxelInit(target_cloud, filtered_target_cloud);
  voxelInit2(source_cloud, filtered_source_cloud);

  // 初始估计
  Eigen::AngleAxisf init_rotation (0, Eigen::Vector3f::UnitZ ());
  Eigen::Translation3f init_translation (0, 0, 0);
  // Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();   

  /*******************************ICP匹配*************************************/
  std::cout << "------------------------------------" << std::endl;
  setIcpParams();
  icp.setInputTarget(filtered_target_cloud); 
  icp.setInputSource(filtered_source_cloud); 

  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
  TicToc t2;
  icp.align (*output_cloud2, init_guess); 
  std::cout << "ICP has converged:" << icp.hasConverged ()
            << " score: " << icp.getFitnessScore () << std::endl;   //(分数越大，配准效果越差)，欧式适合度评分：距离平方和
  std::cout << "配准后的变换矩阵: \n" << icp.getFinalTransformation () <<std::endl;
  std::cout << "icp 耗时: " << t2.toc() << "s" << std::endl;
  pcl::transformPointCloud (*source_cloud, *output_cloud2, icp.getFinalTransformation ());    // 源点云配准到目标点云坐标系下
  std::cout << "------------------------------------" << std::endl;
  /*******************************NDT匹配*************************************/
  setNdtParams();
  ndt.setInputTarget(filtered_target_cloud); 
  ndt.setInputSource(filtered_source_cloud);  

  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  TicToc t;
  ndt.align (*output_cloud, init_guess); 
  std::cout << "NDT has converged:" << ndt.hasConverged ()
            << " score: " << ndt.getFitnessScore () << std::endl;   //(分数越大，配准效果越差)，欧式适合度评分：距离平方和
  std::cout << "配准后的变换矩阵: \n" << ndt.getFinalTransformation () <<std::endl;
  std::cout << "ndt 耗时: " << t.toc() << "s" << std::endl << std::endl << std::endl;
  pcl::transformPointCloud (*source_cloud, *output_cloud, ndt.getFinalTransformation ());    // 源点云配准到目标点云坐标系下
  // pcl::io::savePCDFileASCII ("../src/ndt_transformed.pcd", *output_cloud);
  // pcl::io::savePCDFileBinary ("../src/ndt_transformed.pcd", *output_cloud);


  // 可视化  ndt  icp
  myDisplay(filtered_target_cloud, output_cloud, filtered_target_cloud, output_cloud2);

  return 0;
}

/******************************************************************************/
void setNdtParams() { 
    ndt.setTransformationEpsilon (0.001);    // 两次转换矩阵的最大转换差异， 越小匹配精度越高， 速度越慢
    ndt.setStepSize (1.0);  //More-Thuente搜索算法 最大步长 
    ndt.setResolution (0.95);  //设置 目标点云的体素分辨率
    ndt.setMaximumIterations (35);  //迭代次数  若先满足Eps则会提前终止
} 
 
void setIcpParams() {
    icp.setTransformationEpsilon(0.001);     // 前后两次迭代的转矩阵的最大转换差异   0.001
    icp.setMaxCorrespondenceDistance(1.5);   // 点对之间的最大距离，只有对应点之间的距离小于该设置值的对应点才作为ICP计算的点对
    icp.setEuclideanFitnessEpsilon(0.01);    // 点对的均方误差和
    icp.setMaximumIterations(35);
} 
// 目标点云滤波 64线束
void voxelInit(pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud) {
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setLeafSize (0.55, 0.55, 0.55);
  voxel_filter.setInputCloud (raw_cloud);   
  voxel_filter.filter (*filtered_cloud);                //源点云滤波
  std::cout << "Filtered cloud size: " << filtered_cloud->size () << std::endl;

}
// 源点云滤波
void voxelInit2(pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud) {
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setLeafSize (1.5, 1.5, 1.5);
  voxel_filter.setInputCloud (raw_cloud);   
  voxel_filter.filter (*filtered_cloud);                //源点云滤波
  std::cout << "Filtered cloud size: " << filtered_cloud->size () << std::endl;

}
// 将点云平移、旋转一定姿态
void movePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr& obj_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& out_cloud) {

// T源_目标
	Eigen::AngleAxisf r_vec(3.14/18, Eigen::Vector3f(0,0,1));
  Eigen::Matrix4f cloud_pose = Eigen::Matrix4f::Identity();;
  // cloud_pose.block<3, 3>(0,0) = r_vec.toRotationMatrix();
  cloud_pose(0,3) = 0.5; 
  cloud_pose(1,3) = 0.5;
  cloud_pose(2,3) = 0;

  pcl::transformPointCloud(*obj_cloud, *out_cloud, cloud_pose); 

}

// 传入配准前后的目标点云与源点云
void myDisplay(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, 
                pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud2, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud2 ) {

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_final (new pcl::visualization::PCLVisualizer ("Viewer"));
 
  int v1(0);
  int v2(1);
  viewer_final->createViewPort(0, 0.0, 0.5, 1.0, v1);   // 创建两个窗口
  viewer_final->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
  viewer_final->setBackgroundColor (255, 255, 255, v1);   //背景黑
  viewer_final->setBackgroundColor (255, 255, 255, v2);  

  /***********************窗口1*******************************/
  // 显示目标点云（红）
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color (target_cloud, 255, 0, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (target_cloud, target_color, "target cloud", v1);
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "target cloud");
  // 显示源点云变换到目标点云坐标系下（绿色）
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color (output_cloud, 0, 255, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (output_cloud, output_color, "output cloud", v1);
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "output cloud");


  /***********************窗口2*******************************/
  // 显示目标点云（红）
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color2 (target_cloud2, 255, 0, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (target_cloud2, target_color2, "target cloud2", v2);
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "target cloud2");
  // 显示源点云变换到目标点云坐标系下（绿色）
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color2 (output_cloud2, 0, 255, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (output_cloud2, output_color2, "output cloud2", v2);
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "output cloud2");
  // 创建坐标系
  viewer_final->addCoordinateSystem(1.0);
  viewer_final->initCameraParameters ();
  while (!viewer_final->wasStopped ())
  {
    viewer_final->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }


}
