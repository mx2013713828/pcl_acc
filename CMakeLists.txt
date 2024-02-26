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

# 添加源文件
set(SOURCE_FILES cuda_pcl.cpp cuda_pcl.cu cuda_pcl.h)

## generate dynamic library
# cuda_add_library(cuda_pcl_lib SHARED ${SOURCE_FILES})

# 编译可执行文件
cuda_add_executable(YourExecutable main.cpp cuda_pcl.cpp cuda_pcl.cu)

# 链接CUDA和PCL库
target_link_libraries(YourExecutable ${CUDA_LIBRARIES} ${PCL_LIBRARIES})