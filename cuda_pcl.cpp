#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_pcl.h"
#include <pcl/io/pcd_io.h>
#include <chrono>
bool cuda_transform_points(pcl::PointCloud<PointXYZIRT> &point_cloud, Eigen::Affine3f matrix) // 变换点云
{
    int threads;                  // 线程数
    PointXYZIRT *d_point_cloud; // 点云,设备DEVICE

    float h_m[16]; // 矩阵,主机HOST
    float *d_m;       // 矩阵,设备DEVICE

    cudaError_t err = ::cudaSuccess;
    err = cudaSetDevice(0); // 设置设备
    if (err != ::cudaSuccess)
        return false;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 假设使用设备 0
    threads = prop.maxThreadsPerBlock;
    std::cout << "每个块的最大线程数: " << prop.maxThreadsPerBlock << std::endl;

    h_m[0] = matrix.matrix()(0, 0); // 将矩阵数据复制到主机
    h_m[1] = matrix.matrix()(1, 0);
    h_m[2] = matrix.matrix()(2, 0);
    h_m[3] = matrix.matrix()(3, 0);

    h_m[4] = matrix.matrix()(0, 1);
    h_m[5] = matrix.matrix()(1, 1);
    h_m[6] = matrix.matrix()(2, 1);
    h_m[7] = matrix.matrix()(3, 1);

    h_m[8] = matrix.matrix()(0, 2);
    h_m[9] = matrix.matrix()(1, 2);
    h_m[10] = matrix.matrix()(2, 2);
    h_m[11] = matrix.matrix()(3, 2);

    h_m[12] = matrix.matrix()(0, 3);
    h_m[13] = matrix.matrix()(1, 3);
    h_m[14] = matrix.matrix()(2, 3);
    h_m[15] = matrix.matrix()(3, 3);
    auto start_all = std::chrono::high_resolution_clock::now();

    err = cudaMalloc((void **)&d_m, 16 * sizeof(float)); // 为矩阵分配内存
    if (err != ::cudaSuccess)
        return false;
    auto end_all = std::chrono::high_resolution_clock::now();
    std::cout << "transform test time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count() << " ms" << std::endl;

    err = cudaMemcpy(d_m, h_m, 16 * sizeof(float), cudaMemcpyHostToDevice); // 将矩阵数据从主机复制到设备
    if (err != ::cudaSuccess)
        return false;

    err = cudaMalloc((void **)&d_point_cloud, point_cloud.points.size() * sizeof(PointXYZIRT)); // 为点云分配内存
    if (err != ::cudaSuccess)
        return false;

    err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size() * sizeof(PointXYZIRT), cudaMemcpyHostToDevice); // 将点云数据从主机复制到设备
    if (err != ::cudaSuccess)
        return false;

    auto start_time = std::chrono::high_resolution_clock::now();
    err = cuda_transform_points_kernel_launch(threads, d_point_cloud, point_cloud.points.size(), d_m); // 变换点云,这里面的算法在另一个文件中
    // if (err != ::cudaSuccess)
    //     return false;
    if (err != ::cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "cudaTransformPoints time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
    err = cudaMemcpy(point_cloud.points.data(), d_point_cloud, point_cloud.points.size() * sizeof(PointXYZIRT), cudaMemcpyDeviceToHost); // 将点云数据从设备复制到主机
    if (err != ::cudaSuccess)
        return false;

    err = cudaFree(d_m);
    if (err != ::cudaSuccess)
        return false;

    err = cudaFree(d_point_cloud);
    d_point_cloud = NULL;
    if (err != ::cudaSuccess)
        return false;

    return true;
}

bool cuda_merge_points(int threads,
                          const pcl::PointCloud<PointXYZIRT>::Ptr& cloudA,
                          const pcl::PointCloud<PointXYZIRT>::Ptr& cloudB,
                          pcl::PointCloud<PointXYZIRT>::Ptr& mergedCloud)
{
    PointXYZIRT *d_cloudA = nullptr, *d_cloudB = nullptr, *d_mergedCloud = nullptr;

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "CUDA set device error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    err = cudaMalloc((void **)&d_cloudA, cloudA->size() * sizeof(PointXYZIRT));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_cloudA error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc((void **)&d_cloudB, cloudB->size() * sizeof(PointXYZIRT));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_cloudB error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_cloudA); 
        return false;
    }

    size_t dataSize = (cloudA->size() + cloudB->size()) * sizeof(PointXYZIRT);
    err = cudaMalloc((void **)&d_mergedCloud, dataSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_mergedCloud error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_cloudB); 
        cudaFree(d_cloudA); 
        return false;
    }

    err = cudaMemcpy(d_cloudA, cloudA->points.data(), cloudA->size() * sizeof(PointXYZIRT), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy cloudA to device error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mergedCloud);
        cudaFree(d_cloudB);
        cudaFree(d_cloudA);
        return false;
    }

    err = cudaMemcpy(d_cloudB, cloudB->points.data(), cloudB->size() * sizeof(PointXYZIRT), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy cloudB to device error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mergedCloud);
        cudaFree(d_cloudB);
        cudaFree(d_cloudA);
        return false;
    }
    // if (threads > prop.maxThreadsPerBlock) threads = prop.maxThreadsPerBlock;

    err = cuda_merge_points_kernel_launch(threads, d_cloudA, cloudA->size(),
                          d_cloudB, cloudB->size(),
                          d_mergedCloud);

    if (err != cudaSuccess) {
        std::cerr << "CUDA merge points error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mergedCloud);
        cudaFree(d_cloudB);
        cudaFree(d_cloudA);
        return false;
    }

    if (d_mergedCloud == nullptr) {
        std::cerr << "CUDA error: Device pointer is null." << std::endl;
        cudaFree(d_mergedCloud);
        cudaFree(d_cloudB);
        cudaFree(d_cloudA);
        return false;
    }

    err = cudaMemcpy(mergedCloud->points.data(), d_mergedCloud, dataSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy device to host error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mergedCloud);
        cudaFree(d_cloudB);
        cudaFree(d_cloudA);
        return false;
    }

    cudaFree(d_mergedCloud);
    cudaFree(d_cloudB);
    cudaFree(d_cloudA);
    
    return true;
}


bool cuda_crop_points(int threads,
                      const pcl::PointCloud<PointXYZIRT>::Ptr& input_cloud,
                      pcl::PointCloud<PointXYZIRT>::Ptr& output_cloud,
                      float min_x, float max_x, float min_y, float max_y) {
    PointXYZIRT *d_input_cloud = nullptr, *d_output_cloud = nullptr;
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "CUDA set device error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    int host_output_size = 0; // 定义主机端的 output_size，初始值为 0
    int *d_output_size;
    cudaMalloc(&d_output_size, sizeof(int)); // 在设备端分配内存，用于存储输出大小
    cudaMemcpy(d_output_size, &host_output_size, sizeof(int), cudaMemcpyHostToDevice); // 将主机端的 output_size 复制到设备端

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    err = cudaMalloc((void **)&d_input_cloud, input_cloud->size() * sizeof(PointXYZIRT));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_input_cloud error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc((void **)&d_output_cloud, input_cloud->size() * sizeof(PointXYZIRT));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_output_cloud error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_cloud); 
        return false;
    }

    size_t dataSize = input_cloud->size() * sizeof(PointXYZIRT);
    err = cudaMemcpy(d_input_cloud, input_cloud->points.data(), dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy input_cloud to device error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output_cloud);
        cudaFree(d_input_cloud);
        return false;
    }

    // 调用点云裁剪核函数
    err = cuda_crop_points_kernel_launch(threads, d_input_cloud, input_cloud->size(), d_output_cloud, d_output_size, min_x, max_x, min_y, max_y);
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output_cloud);
        cudaFree(d_input_cloud);
        return false;
    }

    // 将输出大小从设备端复制回主机端
    cudaMemcpy(&host_output_size, d_output_size, sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Output size after crop: " << host_output_size << std::endl;

    // 复制裁剪后的点云数据回主机内存
    output_cloud->points.resize(host_output_size);
    err = cudaMemcpy(output_cloud->points.data(), d_output_cloud, host_output_size * sizeof(PointXYZIRT), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy device to host error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output_cloud);
        cudaFree(d_input_cloud);
        return false;
    }




    cudaFree(d_output_cloud);
    cudaFree(d_input_cloud);
    
    return true;
}


CudaPointCloudManager::CudaPointCloudManager(size_t max_points) 
{
    buffer_size = max_points;
    cudaMalloc(&d_buffer1, max_points * sizeof(PointXYZIRT));
    cudaMalloc(&d_buffer2, max_points * sizeof(PointXYZIRT));
    cudaMalloc(&d_transform_matrix, 16 * sizeof(float));
}

CudaPointCloudManager::~CudaPointCloudManager() 
{
    if(d_buffer1) cudaFree(d_buffer1);
    if(d_buffer2) cudaFree(d_buffer2);
    if(d_transform_matrix) cudaFree(d_transform_matrix);
}

bool CudaPointCloudManager::process_multiple_clouds(
    const std::vector<pcl::PointCloud<PointXYZIRT>::Ptr>& input_clouds,
    const std::vector<Eigen::Matrix4f>& transforms,
    pcl::PointCloud<PointXYZIRT>::Ptr& output_cloud) {
    
    size_t total_points = 0;
    for (const auto& cloud : input_clouds) {
        total_points += cloud->size();
    }
    
    // 一次性处理所有点云
    size_t offset = 0;
    for (size_t i = 0; i < input_clouds.size(); i++) {
        // 复制点云数据到GPU
        cudaError_t err = cudaMemcpy(d_buffer1 + offset, 
                  input_clouds[i]->points.data(), 
                  input_clouds[i]->size() * sizeof(PointXYZIRT),
                  cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy point cloud to GPU: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
                  
        // 准备变换矩阵
        float h_matrix[16];
        h_matrix[0] = transforms[i](0,0); h_matrix[4] = transforms[i](0,1); 
        h_matrix[8] = transforms[i](0,2); h_matrix[12] = transforms[i](0,3);
        h_matrix[1] = transforms[i](1,0); h_matrix[5] = transforms[i](1,1);
        h_matrix[9] = transforms[i](1,2); h_matrix[13] = transforms[i](1,3);
        h_matrix[2] = transforms[i](2,0); h_matrix[6] = transforms[i](2,1);
        h_matrix[10] = transforms[i](2,2); h_matrix[14] = transforms[i](2,3);
        h_matrix[3] = transforms[i](3,0); h_matrix[7] = transforms[i](3,1);
        h_matrix[11] = transforms[i](3,2); h_matrix[15] = transforms[i](3,3);
        
        err = cudaMemcpy(d_transform_matrix, h_matrix, 16 * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy transform matrix to GPU: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // 执行变换
        err = cuda_transform_points_kernel_launch(512, 
            d_buffer1 + offset, 
            input_clouds[i]->size(),
            d_transform_matrix);
        if (err != cudaSuccess) {
            std::cerr << "Failed to transform points: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
            
        offset += input_clouds[i]->size();
    }
    
    // 一次性复制所有结果回host
    output_cloud->points.resize(total_points);
    cudaError_t err = cudaMemcpy(output_cloud->points.data(), 
               d_buffer1,
               total_points * sizeof(PointXYZIRT),
               cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy results back to host: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
               
    return true;
}