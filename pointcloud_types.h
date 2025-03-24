#ifndef _POINTCLOUD_TYPES_BASE_H
#define _POINTCLOUD_TYPES_BASE_H

#include <pcl/point_types.h>

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
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)( double, timestamp, timestamp)(std::uint16_t, ring, ring))

typedef struct
{
    double x;
    double y;
    double z;
} position_t;


typedef struct
{
    double pitch;   //　弧度
    double roll;    //　弧度
    double yaw;     //　弧度
} eulerian_t;


typedef struct 
{
    double x;
    double y;
    double z;
    double w;
} quanternion_t;

typedef struct 
{
    position_t    position;     // 平移
    quanternion_t orientation;  // 四元数
    eulerian_t    eulerian;     // 欧拉角
} pose_t;

#endif