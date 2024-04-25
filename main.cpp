#define PCL_NO_PRECOMPILE
#include <iostream>
#include "cuda_pcl.h"
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

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
    // pcl::PointCloud<PointXYZIRT>::Ptr cloud(new pcl::PointCloud<PointXYZIRT>);
    // if ( (pcl::io::loadPCDFile<PointXYZIRT>("/home/sdlg/Database/SdlgProject/mayufeng/pcl_acc/test1.pcd", *cloud)) == -1 )
    // {
    //   PCL_ERROR("Failed to read test.pcd file\n");
    //   return -1;
    // }

    // Eigen::Affine3f matrix = Eigen::Affine3f::Identity();
    // matrix.translation() = Eigen::Vector3f(5.0, 5.0, 5.0);
    // pcl::PointCloud<PointXYZIRT> output;
    // auto start_transform = std::chrono::system_clock::now();

    // pcl::transformPointCloud(*cloud,output, matrix);
    // auto end_transform = std::chrono::system_clock::now();
    // std::cout << "PCL::Transform took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_transform - start_transform).count() << " ms" << std::endl;
    
    // pcl::io::savePCDFile<PointXYZIRT>("pcl_transformed.pcd", output);
    // auto start_time = std::chrono::system_clock::now();
    // bool result = transformCUDA(*cloud, matrix);
    // auto end_time = std::chrono::system_clock::now();
    // std::cout << "CUDA::Transform took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
    // pcl::io::savePCDFile<PointXYZIRT>("cuda_transformed.pcd", *cloud);


    // pcl::PointCloud<PointXYZIRT>::Ptr mergedCloud(new pcl::PointCloud<PointXYZIRT>());
    // // 读取点云数据
    // pcl::PointCloud<PointXYZIRT>::Ptr testA(new pcl::PointCloud<PointXYZIRT>);
    // if ( (pcl::io::loadPCDFile<PointXYZIRT>("../test1.pcd", *testA)) == -1 )
    // {
    //   PCL_ERROR("Failed to read test1.pcd file\n");
    //   return -1;
    // }    

    // pcl::PointCloud<PointXYZIRT>::Ptr testB(new pcl::PointCloud<PointXYZIRT>);
    // if ( (pcl::io::loadPCDFile<PointXYZIRT>("../test2.pcd", *testB)) == -1 )
    // {
    //   PCL_ERROR("Failed to read test2.pcd file\n");
    //   return -1;
    // }

    // mergedCloud->points.resize(testA->size() + testB->size());
    // // std::cout<<mergedCloud->size()<<std::endl;
    // auto start_merge = std::chrono::system_clock::now();

    // bool success = mergePointCloudsCUDA(256, testA, testB, mergedCloud);

    // if (success) {
    //     // 在这里，mergedCloud 包含了合并后的点云数据
    //     std::cout << "PointCloud merge successful." << std::endl;

    //     // 可以继续处理或显示合并后的点云
    // } else {
    //     std::cerr << "PointCloud merge failed." << std::endl;
    // }

    // //测试处理后点云数值与pcl库操作是否相同
    // // std::cout<<testB->points[63998].ring<<std::endl;
    // mergedCloud->width = mergedCloud->size();
    // mergedCloud->height = 1;
    // auto end_merge = std::chrono::system_clock::now();
    // std::cout << "PCL::merge took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_merge - start_merge).count() << " ms" << std::endl;
    
    // pcl::PointCloud<PointXYZIRT>::Ptr testC(new pcl::PointCloud<PointXYZIRT>());
    // pcl::PointCloud<PointXYZIRT>::Ptr cuda_mergedpts(new pcl::PointCloud<PointXYZIRT>());

    // cuda_mergedpts->points.resize(testA->size() + testC->size());

    // success = cuda_merge_points( 256,testA,testC,cuda_mergedpts);
    // if (success) {
    //     // 在这里，mergedCloud 包含了合并后的点云数据
    //     std::cout << "PointCloud merge successful." << std::endl;

    //     // 可以继续处理或显示合并后的点云
    // } else {
    //     std::cerr << "PointCloud merge failed." << std::endl;
    // }
    // pcl::io::savePCDFile<PointXYZIRT>("../cuda_merged.pcd", *mergedCloud);


    pcl::PointCloud<PointXYZIRT>::Ptr cloud(new pcl::PointCloud<PointXYZIRT>);
    if ( (pcl::io::loadPCDFile<PointXYZIRT>("../test1.pcd", *cloud)) == -1 )
    {
      PCL_ERROR("Failed to read test.pcd file\n");
      return -1;
    }
    pcl::PointCloud<PointXYZIRT>::Ptr output(new pcl::PointCloud<PointXYZIRT>);

    float min_x = 20;
    float min_y = 20;
    float max_x = 100;
    float max_y = 100;
    auto start_crop = std::chrono::system_clock::now();

    bool success = cuda_crop_points(256,
                                    cloud,
                                    output,
                            min_x, max_x, min_y, max_y);

    auto end_crop = std::chrono::system_clock::now();
    std::cout << "PCL::merge took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_crop - start_crop).count() << " ms" << std::endl;
    

    if (success) {
        // 在这里，mergedCloud 包含了合并后的点云数据
        std::cout << "PointCloud crop successful." << std::endl;

        // 可以继续处理或显示合并后的点云
    } else {
        std::cerr << "PointCloud crop failed." << std::endl;
    }
    output->width = output->size();
    output->height = 1;
    pcl::io::savePCDFile<PointXYZIRT>("../cuda_crop.pcd", *output);
    pcl::io::savePCDFile<PointXYZIRT>("../test1_saved.pcd", *cloud);

    return 0; 

}