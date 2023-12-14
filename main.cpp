#include <iostream>
#include "cuda_transform.h"
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <pcl/common/transforms.h>

bool initializeCUDA() {
    cudaError_t err = cudaFree(0); // 使用一个简单的 CUDA 函数来触发初始化
    if (err != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

int main()
{
    if (!initializeCUDA()) {
        return 1; // CUDA 初始化失败，程序终止
    }
      
    // 读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if ( (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/sdlg/Database/SdlgProject/mayufeng/pcl_acc/test1.pcd", *cloud)) == -1 )
    {
      PCL_ERROR("Failed to read test.pcd file\n");
      return -1;
    }

    Eigen::Affine3f matrix = Eigen::Affine3f::Identity();
    matrix.translation() = Eigen::Vector3f(5.0, 5.0, 5.0);
    pcl::PointCloud<pcl::PointXYZ> output;
    auto start_transform = std::chrono::system_clock::now();

    pcl::transformPointCloud(*cloud,output, matrix);
    auto end_transform = std::chrono::system_clock::now();
    std::cout << "PCL::Transform took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_transform - start_transform).count() << " ms" << std::endl;
    
    pcl::io::savePCDFile<pcl::PointXYZ>("transformed_cloud.pcd", *cloud);
    auto start_time = std::chrono::system_clock::now();
    bool result = transform(*cloud, matrix);
    auto end_time = std::chrono::system_clock::now();
    std::cout << "CUDA::Transform took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
    pcl::io::savePCDFile<pcl::PointXYZ>("transformed_cloud.pcd", *cloud);

    return 0;
}