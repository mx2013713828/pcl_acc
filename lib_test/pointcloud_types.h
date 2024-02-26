#ifndef _POINTCLOUD_TYPES_BASE_H
#define _POINTCLOUD_TYPES_BASE_H

#include <pcl/point_types.h>

//ring属性在划分为不同的线号后是不是就用不到了，先把标记位存放在这里
struct PointXYZIRT {
    //添加pcl里xyz
    PCL_ADD_POINT4D   
    float intensity;
    double timestamp;
    uint16_t ring;                   
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
} EIGEN_ALIGN16;                   

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)( double, timestamp, timestamp)(uint16_t, ring, ring))
#endif