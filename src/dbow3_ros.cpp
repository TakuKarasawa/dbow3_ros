#include "dbow3_ros/dbow3_ros.h"

using namespace dbow3;

DBoW3ROS::DBoW3ROS() :
    private_nh_("~")
{
}

void DBoW3ROS::process() { ros::spin(); }

int main(int argc,char** argv)
{
    ros::init(argc,argv,"dbow3_ros");
    DBoW3ROS dbow3_ros;
    dbow3_ros.process();
    return 0;
}
