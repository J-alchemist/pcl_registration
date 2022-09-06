#ifndef __VOXEL_FILTER_H
#define __VOXEL_FILTER_H

#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <Eigen/Dense>
#include <pcl/common/transforms.h>

// 加载点云.pcd文件
int load_pcd_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud) ;

// 点云预处理
void voxel_filter (pcl::PointCloud<pcl::PointXYZ>::Ptr& inut_cloud, 
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud, double leaf_size);

// 移动点云
void move_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& obj_cloud, 
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& out_cloud);

// 显示
void myDisplay1(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud );
void myDisplay2(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
                pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud2, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud2);







#endif

