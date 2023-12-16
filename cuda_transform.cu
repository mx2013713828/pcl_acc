#include "cuda_transform.h"

__global__ void kernel_cudaTransformPoints(PointXYZIRT *d_point_cloud, int number_of_points, float *d_matrix)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x; // 线程索引

    if (ind < number_of_points) // 线程索引小于点云数
    {
        float vSrcVector[3] = {d_point_cloud[ind].x, d_point_cloud[ind].y, d_point_cloud[ind].z};                          // 点云数据
        float vOut[3];                                                                                                      // 点云数据
        vOut[0] = d_matrix[0] * vSrcVector[0] + d_matrix[4] * vSrcVector[1] + d_matrix[8] * vSrcVector[2] + d_matrix[12]; // 矩阵乘法,用于计算点云数据的转换
        vOut[1] = d_matrix[1] * vSrcVector[0] + d_matrix[5] * vSrcVector[1] + d_matrix[9] * vSrcVector[2] + d_matrix[13];
        vOut[2] = d_matrix[2] * vSrcVector[0] + d_matrix[6] * vSrcVector[1] + d_matrix[10] * vSrcVector[2] + d_matrix[14];

        d_point_cloud[ind].x = vOut[0]; // 将转换后的点云数据存储到原来的点云数据中
        d_point_cloud[ind].y = vOut[1];
        d_point_cloud[ind].z = vOut[2];
    }
}

cudaError_t cudaTransformPoints(int threads, PointXYZIRT *d_point_cloud, int number_of_points, float *d_matrix)
{
    kernel_cudaTransformPoints<<<number_of_points / threads + 1, threads>>>(d_point_cloud, number_of_points, d_matrix); // 设置线程块和线程数，并调用kernel来完成transform转换

    cudaDeviceSynchronize(); // 同步
    return cudaGetLastError();
}


// #include "cuda_merge.h"

__global__ void kernel_cudaMergePoints(PointXYZIRT *d_point_cloud1, int num_points1,
                                       PointXYZIRT *d_point_cloud2, int num_points2,
                                       PointXYZIRT *d_merged_point_cloud)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x; // 线程索引

    // 合并点云，每个线程处理一个点
    if (ind < num_points1)
    {
        d_merged_point_cloud[ind] = d_point_cloud1[ind];
    }
    else if (ind < num_points1 + num_points2)
    {
        d_merged_point_cloud[ind] = d_point_cloud2[ind - num_points1];
    }
    // std::cout<<"d_merged_point_cloud->"<<ind<<" : "<<d_merged_point_cloud[ind]<<std::endl;
}

cudaError_t cudaMergePoints(int threads, PointXYZIRT *d_point_cloud1, int num_points1,
                            PointXYZIRT *d_point_cloud2, int num_points2,
                            PointXYZIRT *d_merged_point_cloud)
{
    // 计算并行执行所需的线程块和线程数
    int total_points = num_points1 + num_points2;
    int blocks = (total_points + threads - 1) / threads;

    // 启动核函数，将两个点云合并到一个新的点云中
    kernel_cudaMergePoints<<<blocks, threads>>>(d_point_cloud1, num_points1,
                                                d_point_cloud2, num_points2,
                                                d_merged_point_cloud);

    // 等待所有线程完成
    cudaDeviceSynchronize();

    // 检查CUDA调用是否成功
    return cudaGetLastError();
}