#include "lidar_utility.h"

Eigen::Matrix4f pose_to_matrix(pose_t pose)
{
    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    
    // 将roll、pitch、yaw转换为旋转矩阵
    Eigen::AngleAxisf angle_roll(pose.eulerian.roll,   Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf angle_pitch(pose.eulerian.pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf angle_yaw(pose.eulerian.yaw,     Eigen::Vector3f::UnitZ());
    
    rotation.block<3, 3>(0, 0) = (angle_yaw * angle_pitch * angle_roll).matrix();
    
    // 设置平移向量
    rotation(0, 3) = pose.position.x;
    rotation(1, 3) = pose.position.y;
    rotation(2, 3) = pose.position.z;

    return rotation;
};

// 定义一个函数来执行坐标变换
PointCloud::Ptr transform_pointcloud_pose(const PointCloud::Ptr& input_cloud, const pose_t pose, bool apply_sparse, bool apply_cuda) 
{

    int threads = 256;

    // 创建一个仿射变换矩阵，包括平移和旋转
    Eigen::Matrix4f rotation;
    rotation = pose_to_matrix(pose);
    PointCloud::Ptr processed_cloud(new PointCloud());

    if (apply_sparse && apply_cuda) {

        // 降采样
        // pcl::VoxelGrid<PointXYZIRT> sor;
        // sor.setInputCloud(input_cloud);
        // sor.setLeafSize(0.1f, 0.1f, 0.1f);  // 设置降采样的体素大小
        // sor.filter(*processed_cloud);
        // // 应用仿射变换到点云数据
        // Eigen::Affine3f affine_transform(rotation);
        // bool result = cuda_transform_points(*processed_cloud, affine_transform);
        return processed_cloud;

    } else {
        if (apply_cuda) {
            Eigen::Affine3f affine_transform(rotation);
            bool result = cuda_transform_points(*input_cloud, affine_transform);
            if (!result) {
                std::cerr << "PointCloud transform failed." << std::endl;
                return PointCloud::Ptr(new PointCloud());  // 返回空指针或采取其他错误处理措施
            }

        } else {
            pcl::transformPointCloud(*input_cloud, *input_cloud, rotation);
        }
        return input_cloud;
    }
    
}



// 定义一个函数来合并点云数据
PointCloud::Ptr merge_pointclouds(const PointCloud::Ptr& cloud_main, const PointCloud::Ptr& cloud_left, const PointCloud::Ptr& cloud_right, const pose_t pose_left, const pose_t pose_right, const pose_t pose_car, bool apply_sparse, bool apply_cuda) 
{

    int threads = 256;
    // 检查左侧雷达
    if (!cloud_left || cloud_left->empty()) {
        std::cerr << "Error: Left cloud is invalid or empty." << std::endl;
        return PointCloud::Ptr(new PointCloud());  // 返回空的 PointCloud
    }

    // 检查主点云
    if (!cloud_main || cloud_main->empty()) {
        std::cerr << "Error: Main cloud is invalid or empty." << std::endl;
        return PointCloud::Ptr(new PointCloud());  // 返回空的 PointCloud
    }

    // 检查右侧雷达
    if (!cloud_right || cloud_right->empty()) {
        std::cerr << "Error: Right cloud is invalid or empty." << std::endl;
        return PointCloud::Ptr(new PointCloud());  // 返回空的 PointCloud
    }

    // 对左侧雷达的点云数据进行坐标变换，包括平移和旋转
    PointCloud::Ptr transformed_cloud_left  = transform_pointcloud_pose(cloud_left, pose_left, apply_sparse, apply_cuda);

    // 对右侧雷达的点云数据进行坐标变换，包括平移和旋转
    PointCloud::Ptr transformed_cloud_right = transform_pointcloud_pose(cloud_right, pose_right, apply_sparse, apply_cuda);

    // 检查左侧雷达转换后的点云是否为空
    if (transformed_cloud_left->empty()) {
        std::cerr << "Error: Transformed cloud left is empty after transformation." << std::endl;
        return PointCloud::Ptr(new PointCloud());
    }
    // 检查右侧雷达转换后的点云是否为空
    if (transformed_cloud_right->empty()) {
        std::cerr << "Error: Transformed cloud right is empty after transformation." << std::endl;
        return PointCloud::Ptr(new PointCloud());
    }

    // 合并三个雷达
    PointCloud::Ptr temp_merged_cloud(new PointCloud());
    PointCloud::Ptr merged_cloud(new PointCloud());

    temp_merged_cloud->points.resize(transformed_cloud_left->size() + transformed_cloud_right->size());

    if  (apply_cuda) {
        Eigen::Affine3f affine_transform(pose_to_matrix(pose_car));
        bool tmp_success = cuda_merge_points(threads, transformed_cloud_left, transformed_cloud_right, temp_merged_cloud);
        if (!tmp_success) {
            std::cerr << "Temporary PointCloud merge failed." << std::endl;
            return PointCloud::Ptr(new PointCloud());  // 返回空指针或采取其他错误处理措施
        }
        // 合并到主点云中
        merged_cloud->points.resize(cloud_main->size() + temp_merged_cloud->size());
        bool success = cuda_merge_points(threads, cloud_main, temp_merged_cloud, merged_cloud);
    
    } else {
        *merged_cloud = *cloud_main + *transformed_cloud_left;
        *merged_cloud += *transformed_cloud_right;        
    }

    // 增加对车辆坐标系的变换
    Eigen::Matrix4f rotation;
    rotation = pose_to_matrix(pose_car);
    if (apply_cuda) {
        Eigen::Affine3f affine_transform(rotation);
        bool tr_success = cuda_transform_points(*merged_cloud, affine_transform);
        if (!tr_success) {
            std::cerr << "mergedPointCloud transform to car failed." << std::endl;
            return PointCloud::Ptr(new PointCloud());  // 返回空指针或采取其他错误处理措施
        }
        // if (!mergedCloud || mergedCloud->empty()) {
        //     std::cerr << "Error: Merged cloud is invalid or empty." << std::endl;
        //     return PointCloud::Ptr(new PointCloud());  // 或者采取其他错误处理措施
        // }
        return merged_cloud;
    
    } else {
        PointCloud::Ptr transformed_cloud;
        pcl::transformPointCloud(*merged_cloud, *transformed_cloud, rotation);
        return transformed_cloud;
    }

}



// 定义一个函数来合并四个点云数据
PointCloud::Ptr merge_pointclouds_extended( const PointCloud::Ptr& cloud_main, const PointCloud::Ptr& cloud_left, const PointCloud::Ptr& cloud_right, const PointCloud::Ptr& cloud_back,
                                            const pose_t pose_left, const pose_t pose_right, const pose_t pose_back, const pose_t pose_car, bool apply_sparse, bool apply_cuda) 
{
    // 合并左侧、右侧和后侧点云
    PointCloud::Ptr merged_temp = merge_pointclouds(cloud_main, cloud_left, cloud_right, pose_left, pose_right, pose_car, apply_sparse, apply_cuda);

    // 检查合并后的临时点云是否有效
    if (!merged_temp || merged_temp->empty()) {
        std::cerr << "Error: Merged temporary cloud is invalid or empty." << std::endl;
        return PointCloud::Ptr(new PointCloud());  // 返回空的 PointCloud
    }

    // 对后侧点云进行坐标变换，将其转换到车体坐标系上
    PointCloud::Ptr transformed_cloud_back_temp = transform_pointcloud_pose(cloud_back, pose_back, apply_sparse, apply_cuda);
    PointCloud::Ptr transformed_cloud_back = transform_pointcloud_pose(transformed_cloud_back_temp, pose_car, apply_sparse, apply_cuda);

    
    if (transformed_cloud_back->empty()) { // 检查转换后的后侧点云是否为空
        std::cerr << "Error: Transformed cloud back is empty after transformation." << std::endl;
        return PointCloud::Ptr(new PointCloud());
    }

    PointCloud::Ptr merged_cloud(new PointCloud());
    
    if (apply_cuda) { // 合并后侧点云到临时点云中
        merged_cloud->points.resize(merged_temp->size() + transformed_cloud_back->size());
        bool tmp_success = cuda_merge_points(256, merged_temp, transformed_cloud_back, merged_cloud);
        if (!tmp_success) {
            std::cerr << "Error: CUDA merge pcd back failed." << std::endl;
            return merged_temp; // 返回合并的主、左、右
        }
    
    } else {
        *merged_cloud = *merged_temp + *transformed_cloud_back;
    }

    
    return merged_cloud; // 返回合并后的点云
}
