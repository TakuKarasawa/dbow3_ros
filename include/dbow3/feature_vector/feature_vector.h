#ifndef FEATURE_VECTOR_H_
#define FEATURE_VECTOR_H_

#include <iostream>
#include <map>
#include <vector>

#include "dbow3/bow_vector/bow_vector.h"

namespace dbow3
{
// Vector of nodes with indexes of local features
class FeatureVector : public std::map<NodeId,std::vector<unsigned int>>
{
public:
	FeatureVector();
	~FeatureVector();
	
	void addFeature(NodeId id,unsigned int i_feature);
	friend std::ostream& operator<<(std::ostream& out,const FeatureVector& v);
    
};
}	// namespace dbow3

#endif	// FEATURE_VECTOR_H_