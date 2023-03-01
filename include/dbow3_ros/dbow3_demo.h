#ifndef DBOW3_DEMO_H_
#define DBOW3_DEMO_H_

// ros
#include <ros/ros.h>

// opencv2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/database/database.h"


namespace dbow3
{
class DBoW3Demo
{
public:
    DBoW3Demo();
    void process();

private:
    void set_detector_mode(std::string mode);
    void create_vocabulary(const std::vector<cv::Mat>& features);
    void create_database(std::vector<cv::Mat>& features);
    std::vector<cv::Mat> load_features();

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

    // buffer
    std::string file_path_;

    // param
    std::string DIR_PATH_;
};
} // namespace dbow3

#endif  // DBOW3_DEMO_H_
