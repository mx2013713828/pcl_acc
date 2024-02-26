cmake_minimum_required(VERSION 3.10)

project(YourProjectName)

# 设置CUDA
find_package(CUDA REQUIRED)
# 设置CUDA架构，替换为你的GPU架构
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_86,code=sm_86)

# 设置PCL
set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.10)
set(PCL_LIBRARY_DIRS /usr/local/lib)
set(PCL_LIBRARIES pcl_common pcl_io pcl_filters)  # 根据你的需求选择需要的PCL库

# 包含CUDA头文件路径
include_directories(${CUDA_INCLUDE_DIRS})
# 包含PCL头文件路径
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


# 编译可执行文件
cuda_add_executable(YourExecutable main.cpp )

# 链接CUDA和PCL库
target_link_libraries(YourExecutable ${CMAKE_CURRENT_SOURCE_DIR}/libcuda_pcl_lib.so ${CUDA_LIBRARIES} ${PCL_LIBRARIES})
# target_link_libraries(your_executable ${CMAKE_CURRENT_SOURCE_DIR}/libcuda_pcl_lib.so ${PCL_LIBRARIES})
