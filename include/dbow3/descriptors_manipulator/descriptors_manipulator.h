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
	static void meanValue(const std::vector<cv::Mat>& descriptors,cv::Mat& mean);
	static double distance(const cv::Mat& a,const cv::Mat& b);
	static inline uint32_t distance_8uc1(const cv::Mat& a,const cv::Mat& b);
	static std::string toString(const cv::Mat& a);
	static void fromString(cv::Mat& a,const std::string& s);
	static void toMat32F(const std::vector<cv::Mat>& descriptors,cv::Mat& mat);
	static void toStream(const cv::Mat &m,std::ostream &str);
	static void fromStream(cv::Mat &m,std::istream &str);
	static size_t getDescSizeBytes(const cv::Mat & d){return d.cols* d.elemSize();}

private:
};
}	// namespace dbow3 

#endif	// DESCRIPTORS_MANIPLATOR_H_