#ifndef DESCRIPTORS_MANIPLATOR_H_
#define DESCRIPTORS_MANIPLATOR_H_

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>

namespace dbow3
{
// Class to manipulate descriptors (calculating means, differences and IO routines)
class DescriptorsManipulator
{
public:
    static void mean_value(const std::vector<cv::Mat>& descriptors,cv::Mat& mean);
    static double distance(const cv::Mat& a,const cv::Mat& b);
    static inline uint32_t distance_8uc1(const cv::Mat& a,const cv::Mat& b);
    static std::string to_string(const cv::Mat& a);
    static void from_string(cv::Mat& a,const std::string& s);
    static void to_Mat32F(const std::vector<cv::Mat>& descriptors,cv::Mat& mat);
    static void to_stream(const cv::Mat &m,std::ostream &str);
    static void from_stream(cv::Mat &m,std::istream &str);
    static size_t get_desc_size_bytes(const cv::Mat & d){return d.cols* d.elemSize();}

private:
};

uint32_t DescriptorsManipulator::distance_8uc1(const cv::Mat& a,const cv::Mat& b)
{
    const uint64_t* pa;
    const uint64_t* pb;
    pa = a.ptr<uint64_t>();
    pb = b.ptr<uint64_t>();

    uint64_t v, ret = 0;
    int n = a.cols/sizeof(uint64_t);
    for(size_t i = 0; i < n; i++, pa++, pb++){
        v = *pa ^ *pb;
        v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
        v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) & (uint64_t)~(uint64_t)0/15*3);
        v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
        ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT;
    }
    return ret;
}

}   // namespace dbow3

#endif  // DESCRIPTORS_MANIPLATOR_H_
