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

struct Pose {
    double roll;
    double pitch;
    double yaw;
    double x;
    double y; 
    double z;
    Pose() : roll(0.0), pitch(0.0), yaw(0.0), x(0.0), y(0.0), z(0.0) {}
    Pose(double _roll, double _pitch, double _yaw, double _x, double _y, double _z):roll(_roll), pitch(_pitch), yaw(_yaw), x(_x), y(_y), z(_z) {}
};

// cudaError_t cudaTransformPoints(int threads, pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix);
cudaError_t cuda_transform_points_kernel_launch(int threads, PointXYZIRT *d_point_cloud, int number_of_points, float *d_matrix);

// bool transform(pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Affine3f matrix); // 变换点云
bool cuda_transform_points(pcl::PointCloud<PointXYZIRT> &point_cloud, Eigen::Affine3f matrix); // 变换点云

cudaError_t cuda_merge_points_kernel_launch(int threads, PointXYZIRT *d_point_cloud1, int num_points1,
                            PointXYZIRT *d_point_cloud2, int num_points2,
                            PointXYZIRT *d_merged_point_cloud);

bool cuda_merge_points(int threads,
                          const pcl::PointCloud<PointXYZIRT>::Ptr& cloudA,
                          const pcl::PointCloud<PointXYZIRT>::Ptr& cloudB,
                          pcl::PointCloud<PointXYZIRT>::Ptr& mergedCloud);

cudaError_t cuda_crop_points_kernel_launch(int threads,
                           const PointXYZIRT* d_input_cloud,
                           size_t input_size,
                           PointXYZIRT* d_output_cloud,
                           int *output_size,
                           float min_x, float max_x, float min_y, float max_y);

bool cuda_crop_points(int threads,
                      const pcl::PointCloud<PointXYZIRT>::Ptr& input_cloud,
                      pcl::PointCloud<PointXYZIRT>::Ptr& output_cloud,
                      float min_x, float max_x, float min_y, float max_y);

                      