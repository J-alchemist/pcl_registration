#include "../inc/voxel_filter.h"


int load_pcd_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)  {

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("../src/key_frame_0.pcd", *input_cloud) == -1)     //目标点云
    {
        std::cout << "Couldn't read file key_frame_0.pcd \n" <<std::endl;
        return -1;
    }

    return 0;
}


void voxel_filter (pcl::PointCloud<pcl::PointXYZ>::Ptr& inut_cloud, 
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud, double leaf_size) {

    pcl::VoxelGrid<pcl::PointXYZ> vf;
    vf.setLeafSize (leaf_size, leaf_size, leaf_size);
    vf.setInputCloud (inut_cloud);   
    vf.filter (*filtered_cloud);  

    std::vector<int> indices; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*filtered_cloud, *filtered_cloud, indices);
}

// 将目标点云变换一下作为源点云
// 将点云平移、旋转一定姿态
void move_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& obj_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& out_cloud) {

// T源_目标
	Eigen::AngleAxisf r_vec(3.14/18, Eigen::Vector3f(0,0,1));
    Eigen::Matrix4f cloud_pose = Eigen::Matrix4f::Identity();;
    // cloud_pose.block<3, 3>(0,0) = r_vec.toRotationMatrix();
    cloud_pose(0,3) = 0.5; 
    cloud_pose(1,3) = 0.5;
    cloud_pose(2,3) = 0;

    pcl::transformPointCloud(*obj_cloud, *out_cloud, cloud_pose); 

}

// 显示一种算法配准前后
// 传入配准前后的目标点云与源点云
void myDisplay1(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud ) {

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_final (new pcl::visualization::PCLVisualizer ("Viewer"));
 
  int v1(0);
  viewer_final->createViewPort(0, 0.0, 0.5, 1.0, v1);   // 创建两个窗口
  viewer_final->setBackgroundColor (255, 255, 255, v1);   //背景黑

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

  // 创建坐标系
  viewer_final->addCoordinateSystem(1.0);
  viewer_final->initCameraParameters ();
  while (!viewer_final->wasStopped ())
  {
    viewer_final->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }


}

// 传入配准前后的目标点云与源点云
void myDisplay2(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, 
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