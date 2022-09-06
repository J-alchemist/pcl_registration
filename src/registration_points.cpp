#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include "../inc/tic_toc.h"
#include "../inc/voxel_filter.h"
/*
 * NDT内部将目标点云体素网格化（可设置网格大小），计算每个网格分布，
 * -->通过调整 T目标_源， 将源点云转到目标点云坐标系，计算最佳分布，输出最佳的 T目标_源.
 */
//ndt算法
pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt; 
void setNdtParams() { 
    ndt.setTransformationEpsilon (0.05);    // 两次转换矩阵的最大转换差异， 越小匹配精度越高， 速度越慢
    ndt.setStepSize (0.65);  //More-Thuente搜索算法 最大步长 
    ndt.setResolution (0.95);  //设置 目标点云的体素分辨率
    ndt.setMaximumIterations (35);  //迭代次数  若先满足Eps则会提前终止
} 
//icp算法
pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
void setIcpParams() {
    icp.setTransformationEpsilon(0.001);     // 前后两次迭代的转矩阵的最大转换差异   0.001
    icp.setMaxCorrespondenceDistance(1.0);   // 点对之间的最大距离，只有对应点之间的距离小于该设置值的对应点才作为ICP计算的点对
    icp.setEuclideanFitnessEpsilon(0.01);    // 点对的均方误差和
    icp.setMaximumIterations(35);
} 

// main
int main (int argc, char** argv) { 
  /*******************************通用处理*************************************/

  //加载 目标点云：target_cloud   源点云：source_cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  load_pcd_points(target_cloud); 
  std::cout << "Load target cloud size: " << target_cloud->size () << std::endl;
  // 源点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  move_points(target_cloud, source_cloud);     // 将目标点云做一点变换得到源点云
  std::cout << "Load source cloud size: " << source_cloud->size () << std::endl;
  // 体素滤波器 
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud (new pcl::PointCloud<pcl::PointXYZ>);    
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source_cloud (new pcl::PointCloud<pcl::PointXYZ>);    
  voxel_filter(target_cloud, filtered_target_cloud, 0.5);
  voxel_filter(source_cloud, filtered_source_cloud, 1.5);
 std::cout << "Load filtered_target_cloud size: " << filtered_target_cloud->size () << std::endl;
 std::cout << "Load filtered_source_cloud size: " << filtered_source_cloud->size () << std::endl;
  // 初始估计
  // Eigen::AngleAxisf init_rotation (0, Eigen::Vector3f::UnitZ ());
  // Eigen::Translation3f init_translation (0, 0, 0);
  // Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();
  // Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();   

  /*******************************ICP匹配*************************************/
  std::cout << "------------------------------------" << std::endl;
  TicToc t2;
  setIcpParams();
  icp.setInputTarget(filtered_target_cloud); 
  icp.setInputSource(filtered_source_cloud); 

  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
  icp.align (*output_cloud2); 
  // icp.align (*output_cloud2, init_guess); 
  std::cout << "ICP has converged:" << icp.hasConverged ()
            << " score: " << icp.getFitnessScore () << std::endl;   //(分数越大，配准效果越差)，欧式适合度评分：距离平方和
  std::cout << "配准后的变换矩阵: \n" << icp.getFinalTransformation () <<std::endl;
  std::cout << "icp 耗时: " << t2.toc() << "s" << std::endl;
  pcl::transformPointCloud (*source_cloud, *output_cloud2, icp.getFinalTransformation ());    // 源点云配准到目标点云坐标系下
  std::cout << "------------------------------------" << std::endl;
  /*******************************NDT匹配*************************************/
  TicToc t;
  setNdtParams();
  ndt.setInputTarget(filtered_target_cloud); 
  ndt.setInputSource(filtered_source_cloud);  

  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  ndt.align (*output_cloud); 
  std::cout << "NDT has converged:" << ndt.hasConverged ()
            << " score: " << ndt.getFitnessScore () << std::endl;   //(分数越大，配准效果越差)，欧式适合度评分：距离平方和
  std::cout << "配准后的变换矩阵: \n" << ndt.getFinalTransformation () <<std::endl;
  std::cout << "ndt 耗时: " << t.toc() << "s" << std::endl << std::endl << std::endl;
  pcl::transformPointCloud (*source_cloud, *output_cloud, ndt.getFinalTransformation ());    // 源点云配准到目标点云坐标系下
  // pcl::io::savePCDFileASCII ("../src/ndt_transformed.pcd", *output_cloud);
  // pcl::io::savePCDFileBinary ("../src/ndt_transformed.pcd", *output_cloud);

  // 可视化  ndt  icp
  //myDisplay2(filtered_target_cloud, output_cloud, filtered_target_cloud, output_cloud2);

  return 0;
}

/******************************************************************************/

