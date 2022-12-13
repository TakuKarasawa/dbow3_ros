#ifndef DBOW3_ROS_H_
#define DBOW3_ROS_H_

#include <ros/ros.h>

namespace dbow3
{
class DBoW3ROS
{
public:
    DBoW3ROS();
    void process();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

};
}

#endif  // DBOW3_ROS_H_
