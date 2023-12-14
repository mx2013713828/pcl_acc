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

cudaError_t cudaTransformPoints(int threads, pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix);

bool transform(pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Affine3f matrix); // 变换点云
