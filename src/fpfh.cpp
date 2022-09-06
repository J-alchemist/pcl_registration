#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>
#include <pcl/features/fpfh.h>   
#include <pcl/visualization/pcl_plotter.h> 
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include "../inc/tic_toc.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ia_ransac.h>   //sac_ia算法
#include "../inc/voxel_filter.h"

using namespace pcl;

void fpfh_normal_est(pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud, 
                     pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals, 
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs);

int main(int argc, char **argv) {
       //读取点云  kitti的64线 velo
       // 点云1
       pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	   load_pcd_points(input_cloud); 
       std::cout << "input cloud size: " << input_cloud->size () << std::endl;
       
       // 点云2
       pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  	   move_points(input_cloud, source_cloud); 
       std::cout << "source cloud size: " << source_cloud->size () << std::endl;  		
        
        // 点云滤波
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
   		pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  		voxel_filter(input_cloud, input_filtered_cloud, 0.5);
  		voxel_filter(source_cloud, source_filtered_cloud, 0.5);   		
       std::cout << " Filtered input cloud size: " << input_filtered_cloud->size () << std::endl;
       std::cout << "Filtered source cloud size: " << source_filtered_cloud->size () << std::endl;   

       TicToc t;
       /*******估计两帧点云的描述子********/
       pcl::PointCloud<pcl::FPFHSignature33>::Ptr  fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());
       pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
       pcl::PointCloud<pcl::FPFHSignature33>::Ptr  fpfhs2(new pcl::PointCloud<pcl::FPFHSignature33>());
       pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>());
       // 计算fpfh特征
       fpfh_normal_est(input_filtered_cloud, cloud_normals, fpfhs);
       fpfh_normal_est(source_filtered_cloud, cloud_normals2, fpfhs2);
		double temp_t =  t.toc();
		std::cout << "fpfh耗时： " << temp_t << "s" << std::endl; 
       // 匹配
       pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
       pcl::PointCloud<pcl::PointXYZ>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZ>());   //对齐后的点云
       sac_ia.setInputTarget(input_filtered_cloud);
       sac_ia.setTargetFeatures(fpfhs);
       sac_ia.setInputSource(source_filtered_cloud);
       sac_ia.setSourceFeatures(fpfhs2);
       //sac_ia.setNumberOfSamples(20);  //设置每次迭代计算中使用的样本数量（可省）,可节省时间
       //sac_ia.setCorrespondenceRandomness(50); //设置计算协方差时选择多少近邻点，该值越大，协防差越精确，但是计算效率越低.(可省)
       sac_ia.align(*sac_result);
       Eigen::Matrix4d T(4,4);
       T= sac_ia.getFinalTransformation().cast<double>();
       std::cout << "转换矩阵: \n" << T << std::endl;
       std::cout << "sac_ia耗时： " << t.toc()-temp_t << "s" << std::endl; 

       // 直方图 显示某点的fhfh特征
       pcl::visualization::PCLPlotter plotter;
       plotter.addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs,"fpfh", 100);    // 横轴长度   
       // 点云 可视化
       pcl::visualization::PCLVisualizer viewer("Fpfh Viewer");
       viewer.setBackgroundColor(0.0, 0.0, 0.0);
       			// 显示法向量
       			       
       viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(input_filtered_cloud, cloud_normals, 10, 0.4, "normals");//每10个点显示一个法线，长度为0.4
       viewer.addPointCloud(input_filtered_cloud,"inut_cloud");
viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,1,0,0.5, "inut_cloud");	// 颜色参数3个: 0-1范围
       while (!viewer.wasStopped())
       {
               plotter.plot();
               viewer.spinOnce(100);
       }
       return 0;
}

// #include <pcl/features/normal_3d_omp.h>//使用OMP需要添加的头文件
// #include <pcl/features/fpfh_omp.h> //fpfh加速计算的omp(多核并行计算)
// 见博客
void fpfh_normal_est(pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud,
                     pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals,
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs) {
       //估计法线
       //构建kd树搜索，速度快
       pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal;
       pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
       normal.setInputCloud(filtered_cloud);
       normal.setSearchMethod(tree);
       normal.setRadiusSearch(0.6);      // 查询点周领域搜索半径0.6
       normal.compute(*cloud_normals);   //计算法线

        //构建fpfh对象，传入 点云，法线，计算出fpfh特征
        //构建kd树搜索，速度快，FPFHSignature33： 3个角特征 × 11个直方图
       pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
       pcl::search::KdTree<PointXYZ>::Ptr tree_fpfh(new pcl::search::KdTree<pcl::PointXYZ>);
       fpfh.setInputCloud(filtered_cloud);
       fpfh.setInputNormals(cloud_normals);
       fpfh.setSearchMethod(tree_fpfh); 
        //    fpfh特征
       fpfh.setRadiusSearch(0.8);       // 使用半径必须大于估计法线时使用的半径  
       fpfh.compute(*fpfhs);
       // reset
       tree.reset(new pcl::search::KdTree<pcl::PointXYZ>());
       tree_fpfh.reset(new pcl::search::KdTree<pcl::PointXYZ>());
}
