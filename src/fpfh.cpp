#include <iostream>
#include <ctime>
#include <cmath>
#include <deque>
#include <vector> 
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

using namespace pcl;        // 域名是可以叠加的
//using namespace std;

// 结构体c++定义变量可以省略struct
struct corrs {
       int source_index;                  // 源点云中点的索引
       std::vector<int> target_index;     // 源点云最近邻k个的目标点云索引
};

void fpfh_normal_est(pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud, 
                     pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals, 
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs);
void pcl_sac_ia (pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33>& sac_ia, 
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& input_filtered_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& source_filtered_cloud,
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs,
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs2,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& sac_result,
                     Eigen::Matrix4d& T);
int calc_cloud_dist ( pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                            std::vector<int>& idx_vec, 
                            int valid_dist );
void icp_3d23d_est (std::vector<Eigen::Vector3d>& v1, std::vector<Eigen::Vector3d>& v2,
                            Eigen::Matrix4d& T);
double compute_error(pcl::PointCloud<pcl::PointXYZ>::Ptr& source, 
                            pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
                                   Eigen::Matrix4d& T);
void my_sac_ia ( pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, 
                 pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud, 
                 pcl::PointCloud<pcl::FPFHSignature33>::Ptr& target_fpfhs,
                 pcl::PointCloud<pcl::FPFHSignature33>::Ptr& source_fpfhs,
                 int sample_size, Eigen::Matrix4d& T );

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
       Eigen::Matrix4d T(4,4);
       // pcl_sac_ia(sac_ia, input_filtered_cloud, source_filtered_cloud, fpfhs, fpfhs2, sac_result, T);
       // std::cout << "转换矩阵: \n" << T << std::endl;
       // std::cout << "sac_ia耗时： " << t.toc()-temp_t << "s" << std::endl; 
       // temp_t = t.toc()-temp_t;

       T = Eigen::Matrix4d::Identity();
	my_sac_ia(input_filtered_cloud, source_filtered_cloud, fpfhs, fpfhs2, 3, T);
       std::cout << "我的变换矩阵: \n" << T << std::endl;
       std::cout << "我的sac_ia耗时： " << t.toc()-temp_t << "s" << std::endl; 

//        // 直方图 显示某点的fhfh特征
//        pcl::visualization::PCLPlotter plotter;
//        plotter.addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs,"fpfh", 100);    // 横轴长度   
//        // 点云 可视化
//        pcl::visualization::PCLVisualizer viewer("Fpfh Viewer");
//        viewer.setBackgroundColor(0.0, 0.0, 0.0);
//        			// 显示法向量
       			       
//        viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(input_filtered_cloud, cloud_normals, 10, 0.4, "normals");//每10个点显示一个法线，长度为0.4
//        viewer.addPointCloud(input_filtered_cloud,"inut_cloud");
// viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,1,0,0.5, "inut_cloud");	// 颜色参数3个: 0-1范围
//        while (!viewer.wasStopped())
//        {
//                plotter.plot();
//                viewer.spinOnce(100);
//        }
       return 0;
}

// #include <pcl/features/normal_3d_omp.h>//使用OMP需要添加的头文件
// #include <pcl/features/fpfh_omp.h> //fpfh加速计算的omp(多核并行计算)
// 见我的博客
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


void pcl_sac_ia (pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33>& sac_ia,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& input_filtered_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& source_filtered_cloud,
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs,
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs2,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& sac_result,
                     Eigen::Matrix4d& T)         {
       
       sac_ia.setInputTarget(input_filtered_cloud);
       sac_ia.setTargetFeatures(fpfhs);
       sac_ia.setInputSource(source_filtered_cloud);
       sac_ia.setSourceFeatures(fpfhs2);
       //sac_ia.setNumberOfSamples(20);  //设置每次迭代计算中使用的样本数量（可省），
       //sac_ia.setCorrespondenceRandomness(50); //设置计算协方差时选择多少近邻点，该值越大，协防差越精确，但是计算效率越低, K近邻
       sac_ia.align(*sac_result, Eigen::Matrix4f::Identity());        // 给初值就是先将源点云变换之后再匹配
       T= sac_ia.getFinalTransformation().cast<double>();
}


// 目标点云特征target_fpfhs，源点云特征source_fpfhs，采样点大小sample_size，输出变换矩阵T
void my_sac_ia ( pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, 
                 pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud, 
                 pcl::PointCloud<pcl::FPFHSignature33>::Ptr& target_fpfhs,
                 pcl::PointCloud<pcl::FPFHSignature33>::Ptr& source_fpfhs,
                 int sample_size, Eigen::Matrix4d& T) {  

       // std::cout << "source->at: " << target_cloud->at(0) << std::endl;
       // (26.5125,5.4335,-11.2905) ==> 类型pcl::PointXYZ
// 1-查找  为每个源特征查找对应 k个近邻的目标特征  
       int K = 10;   // 设置K的个数  
       std::vector<int> idx(K);  // 创建一个 搜索后 保存 的 点的索引值 的 向量
       std::vector<float> squaredDis(K);     // 创建一个 搜索后 保存 的 点的距离平方值 的 向量     
       size_t source_size = source_fpfhs->size();
       std::vector<corrs> matchSet(source_size);        // 作为左值，必须先初始化

       pcl::search::KdTree<pcl::FPFHSignature33>::Ptr fpfh_tree(new pcl::search::KdTree<pcl::FPFHSignature33>());
       fpfh_tree->setInputCloud(target_fpfhs);
       // 通过描述子找最近邻
       for (int i=0; i<source_size; ++i) { 
              // source_fpfhs[i] 与 target_fpfhs最近的K个
              fpfh_tree->nearestKSearch(*source_fpfhs, i, K, idx, squaredDis);
              matchSet.at(i).source_index = i;
              matchSet.at(i).target_index = idx; 
       } 

// 2-在matchSet中随机选择匹配点
       int sampleSize = sample_size;
       std::vector<int> final_target_matches; // 存储点对
       std::vector<int> final_source_matches; 

       srand((unsigned int)time(NULL)); 

   while (1) {
       for(;;) { 
              int temp_rand_idx = (rand() % (source_size-0)) + 0;           // [0, source_size)
              final_source_matches.push_back(matchSet[temp_rand_idx].source_index);
              if (calc_cloud_dist(source_cloud, final_source_matches, 100) == sampleSize) {
                     std::cout << "step1 over!" << std::endl;

                     for (int i = 0; i<final_source_matches.size(); ++i) {

                            std::cout <<     final_source_matches.at(i)  << std::endl;
                            std::cout << "_________________" << std::endl;
                     }
                     break;
              }
       }

       // 为选出的每个源点云的点，在目标点云的k近邻中寻找对应的一个匹配点
       for (int i=0; i<sampleSize; i++) { 

              std::vector<int> source2target_idx_vec( matchSet[final_source_matches.at(i)].target_index );

              int temp_rand_idx = (rand() % (K-0)) + 0; 
              final_target_matches.push_back( source2target_idx_vec.at(temp_rand_idx) );
              source2target_idx_vec.clear();
       }
       std::cout << "step2 over!" << std::endl;
       for (int i = 0; i<final_target_matches.size(); ++i) {

              std::cout << final_target_matches.at(i)  << std::endl;
              std::cout << "_________________" << std::endl;
       }  

// 3-求解：R，t
       std::vector<Eigen::Vector3d>  source_points, target_points;
       T = Eigen::Matrix4d::Identity();
       for (int i=0; i< sampleSize; i++) {
              target_points.push_back( Eigen::Vector3d( target_cloud->points[final_target_matches.at(i)].x, 
                                                        target_cloud->points[final_target_matches.at(i)].y, 
                                                        target_cloud->points[final_target_matches.at(i)].z) );
              source_points.push_back( Eigen::Vector3d(source_cloud->points[final_source_matches.at(i)].x, 
                                                        source_cloud->points[final_source_matches.at(i)].y, 
                                                        source_cloud->points[final_source_matches.at(i)].z) );
              std::cout << "chose source points: \n" << source_points.at(i) << std::endl;
              std::cout << "chose target points: \n" << target_points.at(i) << std::endl;
       }
       
       icp_3d23d_est(target_points, source_points, T);    
// 误差计算：两次变化矩阵的差值
       int error = 0.0;
       error = compute_error(source_cloud, target_cloud, T);
       if ( error < 2) {
              std::cout << "calculate finished！！！" << std::endl;
              break;
       }else {
              std::cout << "now error:\n" << error << std::endl;
       }
   }

}

// 某个点云
// 某些点的index
// valid_dist最小距离
// 返回合法点数
int calc_cloud_dist ( pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                            std::vector<int>& idx_vec, 
                            int valid_dist ) {                      
       bool isValid = true;
       int dist = 0;
       int len = idx_vec.size();
       if (len < 2)
              return len;

       // 校验取出来的点是否符合要求 源点云选出来的点之间要有一定距离
       for (int i = 0; i<len-1; ) {
              isValid = true;
              for (int j = i+1; j<len; j++) { 
                     // 欧氏距离
                     dist = sqrt(  pow(cloud->points[idx_vec.at(i)].x - cloud->points[idx_vec.at(j)].x, 2) + 
                                   pow(cloud->points[idx_vec.at(i)].y - cloud->points[idx_vec.at(j)].y, 2) +
                                   pow(cloud->points[idx_vec.at(i)].z - cloud->points[idx_vec.at(j)].z, 2) );

                     if (dist<valid_dist) {      // 不合法
                            isValid = false;
                            break;
                     }
              }

              if (isValid==true && len!=1) {
                     ++i;
              }else if (isValid==false && len!=1) {
                     idx_vec.erase( idx_vec.begin() + i );
                     len = idx_vec.size();
                     return len; 
              }

              len = idx_vec.size();
              if (len <2)
                     return len;
       }

       return len;
} 

// 输出v2到v1的坐标变换
void icp_3d23d_est (std::vector<Eigen::Vector3d>& v1, std::vector<Eigen::Vector3d>& v2,
                            Eigen::Matrix4d& T) {  
       int len = v1.size();
       Eigen::Vector3d v1_center(0,0,0), v2_center(0,0,0);
       // std::vector<Eigen::Vector3d> aft_v1, aft_v2;
       Eigen::Matrix3d W = Eigen::Matrix3d::Zero();     // 用于SVD分解

       if (len!=v2.size())
              return; 
       // 计算两组点的质心
       for (int i = 0; i<len; ++i) {
              v1_center += v1.at(i);
              v2_center += v2.at(i);
       }
       v1_center /= len; 
       v2_center /= len; 

       // 两组点的每个点到质心的坐标（去质心坐标） &&   aft_v1 * aft_v1^T  
       // aft_v1.clear(); aft_v2.clear();
       for (int i = 0; i<len; ++i) {
              // aft_v1.push_back(v1.at(i)-v1_center);
              // aft_v2.push_back(v2.at(i)-v2_center);

              // 计算aft_v1 * aft_v2^T  
              W += (v1.at(i)-v1_center) * (v2.at(i)-v2_center).transpose();  // 3x3
       }

       // SVD方法分解W
       // U = svd.matrixU();
       // V = svd.matrixV();
       // A = svd.singularValues();       // 对角矩阵的对角线元素
       Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
       Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();        // R
       Eigen::Vector3d t = v1_center - R * v2_center;          // t
       T.block<3,3>(0,0) = R; 
       T.block<3,1>(0,3) = t; 
}


double compute_error(pcl::PointCloud<pcl::PointXYZ>::Ptr& source, 
                            pcl::PointCloud<pcl::PointXYZ>::Ptr& target, 
                                   Eigen::Matrix4d& T) {
       float error = 0;
       int cnt = 0;
       
       pcl::PointCloud<pcl::PointXYZ>::Ptr aft_source(new pcl::PointCloud<pcl::PointXYZ>());
       pcl::transformPointCloud(*source, *aft_source, T.cast<float>());

       pcl::KdTreeFLANN<pcl::PointXYZ> tree;
       tree.setInputCloud(target);

       std::vector<int> nn_index(1);
       std::vector<float> nn_distance(1);

       size_t len = aft_source->size();
       for(int i = 0; i < len; ++i){
              tree.nearestKSearch(*aft_source, i, 1, nn_index, nn_distance);
              if(nn_distance.at(0) > 2.0) 
                     continue;     // 对于距离太远的点，则将其排除误差，此处需要结合点云分辨率设定阈值
              error += nn_distance.at(0);
              cnt++;
       }

       return error / cnt;
}

/*
// 计算两次变换矩阵的误差
void compute_error(Eigen::Matrix4d& T) { 
       double err_r = 0.0, err_t=0.0;
       static std::deque<Eigen::Matrix4d> T_set;
       T_set.push_back(T);

       if (T_set.size() < 2) 
              return;
       while(T_set.size() > 2)
              T_set.pop_front();

       // 旋转误差
       Eigen::Matrix3d R_last =  T_set.front().block<3,3>(0,0);
       Eigen::Matrix3d R_cur =  T_set.back().block<3,3>(0,0);

       // 平移误差
       Eigen::Vector3d t_last = T_set.front().block<3,1>(0,3);
       Eigen::Vector3d t_cur = T_set.back().block<3,1>(0,3);

       // 如何计算
}
*/