#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
// #include <pcl/filters/merge.h>

int main()
{
    // 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("test1.pcd", *cloud1);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("transformed_cloud.pcd", *cloud2);
    
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    *merged_cloud = *cloud1 + *cloud2;

    // 保存合并后的pcd文件
    pcl::io::savePCDFile("merged_file.pcd", *merged_cloud);
    
    return 0;
}