#include "cuda_transform.h"

__global__ void kernel_cudaTransformPoints(pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix)
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

cudaError_t cudaTransformPoints(int threads, pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix)
{
    kernel_cudaTransformPoints<<<number_of_points / threads + 1, threads>>>(d_point_cloud, number_of_points, d_matrix); // 设置线程块和线程数，并调用kernel来完成transform转换

    cudaDeviceSynchronize(); // 同步
    return cudaGetLastError();
}


