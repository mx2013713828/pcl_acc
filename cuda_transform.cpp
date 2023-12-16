#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_transform.h"
#include <pcl/io/pcd_io.h>
#include <chrono>
bool transform(pcl::PointCloud<PointXYZIRT> &point_cloud, Eigen::Affine3f matrix) // 变换点云
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
    err = cudaTransformPoints(threads, d_point_cloud, point_cloud.points.size(), d_m); // 变换点云,这里面的算法在另一个文件中
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

bool mergePointCloudsCUDA(int threads,
                          const pcl::PointCloud<PointXYZIRT>::Ptr& cloudA,
                          const pcl::PointCloud<PointXYZIRT>::Ptr& cloudB,
                          pcl::PointCloud<PointXYZIRT>::Ptr& mergedCloud)
{
    PointXYZIRT *d_cloudA, *d_cloudB, *d_mergedCloud;

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess)
        return false;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    threads = prop.maxThreadsPerBlock;

    err = cudaMalloc((void **)&d_cloudA, cloudA->size() * sizeof(PointXYZIRT));
    if (err != cudaSuccess)
        return false;

    err = cudaMalloc((void **)&d_cloudB, cloudB->size() * sizeof(PointXYZIRT));
    if (err != cudaSuccess)
        return false;

    size_t dataSize = (cloudA->size() + cloudB->size()) * sizeof(PointXYZIRT);
    err = cudaMalloc((void **)&d_mergedCloud, dataSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA 错误: " << cudaGetErrorString(err) << " 在为 d_mergedCloud 进行 cudaMalloc 时" << std::endl;
        return false;
    }


    err = cudaMemcpy(d_cloudA, cloudA->points.data(), cloudA->size() * sizeof(PointXYZIRT), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return false;

    err = cudaMemcpy(d_cloudB, cloudB->points.data(), cloudB->size() * sizeof(PointXYZIRT), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return false;

    auto start_time = std::chrono::high_resolution_clock::now();
    err = cudaMergePoints(threads, d_cloudA, cloudA->size(),
                          d_cloudB, cloudB->size(),
                          d_mergedCloud);

    if (err != cudaSuccess) {
        std::cerr << "CUDA 错误: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "cudaMergePoints 执行时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " 毫秒" << std::endl;
    
    if (d_mergedCloud == nullptr) {
        std::cerr << "CUDA error: Device pointer is null." << std::endl;
        return false;
    }


    err = cudaMemcpy(mergedCloud->points.data(), d_mergedCloud,
                     dataSize,
                     cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
   {
        std::cerr << "CUDA cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    err = cudaFree(d_cloudA);
    if (err != cudaSuccess)
        return false;

    err = cudaFree(d_cloudB);
    if (err != cudaSuccess)
        return false;

    err = cudaFree(d_mergedCloud);
    if (err != cudaSuccess)
        return false;

    return true;
}
