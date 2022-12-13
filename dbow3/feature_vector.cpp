#include "dbow3/feature_vector/feature_vector.h"

using namespace dbow3;

FeatureVector::FeatureVector() {}

FeatureVector::~FeatureVector() {}

void FeatureVector::addFeature(NodeId id,unsigned int i_feature)
{
	FeatureVector::iterator vit = this->lower_bound(id);
	if(vit != this->end() && vit->first == id) vit->second.emplace_back(i_feature);
	else{
    	vit = this->insert(vit,FeatureVector::value_type(id,std::vector<unsigned int>()));
    	vit->second.emplace_back(i_feature);
  	}
}

std::ostream& operator<<(std::ostream& out,const FeatureVector& v)
{
	if(!v.empty()){
		FeatureVector::const_iterator vit = v.begin();
		const std::vector<unsigned int>* f = &vit->second;
		
		out << "<" << vit->first << ": [";
    	if(!f->empty()) out << (*f)[0];
    	for(unsigned int i = 1; i < f->size(); i++) out << ", " << (*f)[i];
		out << "]>";
		
		for(++vit; vit != v.end(); vit++){
			f = &vit->second;
      		out << ", <" << vit->first << ": [";
      		if(!f->empty()) out << (*f)[0];
      		for(unsigned int i = 1; i < f->size(); i++) out << ", " << (*f)[i];
			out << "]>";
    	}
  	}
	return out;  
}