#include "cuda_pcl.h"

__global__ void cuda_transform_points_kernel(PointXYZIRT *d_point_cloud, int number_of_points, float *d_matrix)
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

cudaError_t cuda_transform_points_kernel_launch(int threads, PointXYZIRT *d_point_cloud, int number_of_points, float *d_matrix)
{
    cuda_transform_points_kernel<<<number_of_points / threads + 1, threads>>>(d_point_cloud, number_of_points, d_matrix); // 设置线程块和线程数，并调用kernel来完成transform转换

    cudaDeviceSynchronize(); // 同步
    return cudaGetLastError();
}


// #include "cuda_merge.h"

__global__ void cuda_merge_points_kernel(PointXYZIRT *d_point_cloud1, int num_points1,
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

cudaError_t cuda_merge_points_kernel_launch(int threads, PointXYZIRT *d_point_cloud1, int num_points1,
                            PointXYZIRT *d_point_cloud2, int num_points2,
                            PointXYZIRT *d_merged_point_cloud)
{
    // 计算并行执行所需的线程块和线程数
    int total_points = num_points1 + num_points2;
    int blocks = (total_points + threads - 1) / threads;

    // 启动核函数，将两个点云合并到一个新的点云中
    cuda_merge_points_kernel<<<blocks, threads>>>(d_point_cloud1, num_points1,
                                                d_point_cloud2, num_points2,
                                                d_merged_point_cloud);

    // 等待所有线程完成
    cudaDeviceSynchronize();

    // 检查CUDA调用是否成功
    return cudaGetLastError();
}


// CUDA 核函数，用于裁剪点云
__global__ void cuda_crop_points_kernel(const PointXYZIRT* d_input_cloud, PointXYZIRT* d_output_cloud, size_t size, float min_x, float max_x, float min_y, float max_y, int* output_size) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {

        PointXYZIRT point = d_input_cloud[tid];
        if (point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y) {
            int output_index = atomicAdd(output_size, 1); // 原子操作：增加输出大小

            d_output_cloud[output_index] = point; // 将点复制到输出点云中
        }
    }
}


cudaError_t cuda_crop_points_kernel_launch(int threads,
                                           const PointXYZIRT* d_input_cloud,
                                           size_t input_size,
                                           PointXYZIRT* d_output_cloud,
                                           int* d_output_size,
                                           float min_x, float max_x, float min_y, float max_y
                                           ) 
{
    // 计算并行执行所需的线程块和线程数
    int blocks = (input_size + threads - 1) / threads;
    // 启动核函数，裁剪点云
    cuda_crop_points_kernel<<<blocks, threads>>>(d_input_cloud, d_output_cloud, 
                                                 input_size, 
                                                 min_x, 
                                                 max_x, 
                                                 min_y, 
                                                 max_y,
                                                 d_output_size);

    // 等待所有线程完成
    cudaDeviceSynchronize();

    // 检查CUDA调用是否成功
    return cudaGetLastError();
}