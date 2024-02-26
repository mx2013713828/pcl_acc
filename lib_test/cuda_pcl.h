#include <cuda_runtime.h> 
#include <iostream> 
#include <pcl/point_types.h>
#include <pcl/io/io.h>

#include <pcl/point_cloud.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <cmath>
#include <Eigen/Dense>

#include "pointcloud_types.h"
// cudaError_t cudaTransformPoints(int threads, pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix);
cudaError_t cudaTransformPoints(int threads, PointXYZIRT *d_point_cloud, int number_of_points, float *d_matrix);

// bool transform(pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Affine3f matrix); // 变换点云
bool transformCUDA(pcl::PointCloud<PointXYZIRT> &point_cloud, Eigen::Affine3f matrix); // 变换点云

cudaError_t cudaMergePoints(int threads, PointXYZIRT *d_point_cloud1, int num_points1,
                            PointXYZIRT *d_point_cloud2, int num_points2,
                            PointXYZIRT *d_merged_point_cloud);
bool mergePointCloudsCUDA(int threads,
                          const pcl::PointCloud<PointXYZIRT>::Ptr& cloudA,
                          const pcl::PointCloud<PointXYZIRT>::Ptr& cloudB,
                          pcl::PointCloud<PointXYZIRT>::Ptr& mergedCloud);