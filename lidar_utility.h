#ifndef LIDAR_UTILITY_H
#define LIDAR_UTILITY_H

#include <iostream>
#include <Eigen/Core>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/concatenate.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


#include "cuda_pcl.h"

typedef pcl::PointCloud<PointXYZIRT> PointCloud;

// 定义一个函数来执行坐标变换
PointCloud::Ptr transform_pointcloud_pose(const PointCloud::Ptr& input_cloud, const pose_t pose, bool apply_sparse = false, bool apply_cuda = true); 

// 定义一个函数来合并点云数据
PointCloud::Ptr merge_pointclouds(const PointCloud::Ptr& cloud_main, const PointCloud::Ptr& cloud_left, const PointCloud::Ptr& cloud_right, const pose_t pose_left, const pose_t pose_right, const pose_t pose_car, bool apply_sparse = false, bool apply_cuda = true); 


// 定义一个函数来合并四个点云数据
PointCloud::Ptr merge_pointclouds_extended(const PointCloud::Ptr& cloud_main, const PointCloud::Ptr& cloud_left, const PointCloud::Ptr& cloud_right, const PointCloud::Ptr& cloud_back,
                                           const pose_t pose_left, const pose_t pose_right, const pose_t pose_back, const pose_t pose_car, bool apply_sparse = false, bool apply_cuda = true);


#endif  // LIDAR_UTILITY_H
