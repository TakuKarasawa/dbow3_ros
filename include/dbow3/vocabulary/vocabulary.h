#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <cassert>
#include <limits>

#include <opencv2/core/core.hpp>

#include "dbow3/feature_vector/feature_vector.h"
#include "dbow3/bow_vector/bow_vector.h"
#include "dbow3/scoring_object/scoring_object.h"
#include "dbow3/descriptors_manipulator/descriptors_manipulator.h"
#include "dbow3/quicklz/quicklz.h"

namespace dbow3 
{
class Vocabulary
{		
public:
	Vocabulary(int k = 10,int L = 5,WeightingType weighting = TF_IDF,ScoringType scoring = L1_NORM);
	Vocabulary(const std::string& file_name);
	Vocabulary(const char* file_name);
  	Vocabulary(std::istream& file_name);
	Vocabulary(const Vocabulary& voc);
	
	virtual ~Vocabulary();
	Vocabulary& operator=(const Vocabulary& voc);

	// create
	virtual void create(const std::vector<cv::Mat>& training_features);
  	virtual void create(const std::vector<std::vector<cv::Mat>>& training_features);
	virtual void create(const std::vector<std::vector<cv::Mat>>& training_features,int k,int L);
	virtual void create(const std::vector<std::vector<cv::Mat>>& training_features,int k,int L,WeightingType weighting,ScoringType scoring);

  	virtual inline unsigned int size() const;
	virtual inline bool empty() const;
	void clear();

	// transform
	virtual void transform(const std::vector<cv::Mat>& features,BowVector& v) const;
	virtual void transform(const cv::Mat& features,BowVector& v) const;
	virtual void transform(const std::vector<cv::Mat>& features,BowVector& v,FeatureVector& fv,int levelsup) const;
	virtual WordId transform(const cv::Mat& feature) const;
	
	double score(const BowVector& a,const BowVector& b) const;
	
	virtual NodeId getParentNode(WordId wid, int levelsup) const;
	void getWordsFromNode(NodeId nid,std::vector<WordId>& words) const;
	inline int getBranchingFactor() const;
	inline int getDepthLevels() const;
	float getEffectiveLevels() const;
	virtual inline cv::Mat getWord(WordId wid) const;
	virtual inline WordValue getWordWeight(WordId wid) const;
  	inline WeightingType getWeightingType() const;
	inline ScoringType getScoringType() const;
	inline void setWeightingType(WeightingType type);
 	void setScoringType(ScoringType type);
  
  	void save(const std::string& file_name,bool binary_compressed = true) const;
	virtual void save(cv::FileStorage& fs,const std::string& name = "vocabulary") const;
	void load(const std::string& file_name);
  	bool load(std::istream& stream);
  	virtual void load(const cv::FileStorage& fs,const std::string& name = "vocabulary");

  	virtual int stopWords(double minWeight);
  	int getDescritorSize() const;
  	int getDescritorType() const;

  	void toStream(std::ostream& str,bool compressed=true) const;
  	void fromStream(std::istream& str);

protected:
	// reference to descriptor
  	typedef const cv::Mat pDescriptor;
	
	// Tree node
  	struct Node 
  	{
		Node() : 
			id(0), weight(0), parent(0), word_id(0) {}

    	Node(NodeId _id) : 
			id(_id), weight(0), parent(0), word_id(0) {}
		
		inline bool isLeaf() const { return children.empty(); }

		// Node id
    	NodeId id;
		
		// Weight if the node is a word
    	WordValue weight;
		
		// Children 
    	std::vector<NodeId> children;
		
		// Parent node (undefined in case of root)
    	NodeId parent;
		
		// Node descriptor
    	cv::Mat descriptor;
		
		// Word id if the node is a word
    	WordId word_id;
  	};

protected:
	void createScoringObject();
  	void getFeatures(const std::vector<std::vector<cv::Mat>>& training_features,std::vector<cv::Mat>& features) const;
	
	// transform
	virtual void transform(const cv::Mat& feature,WordId& id,WordValue& weight,NodeId* nid,int levelsup = 0) const;
	virtual void transform(const cv::Mat& feature,WordId& id,WordValue& weight ) const;
	virtual void transform(const cv::Mat& feature,WordId& id) const;
	
	void HKmeansStep(NodeId parent_id,const std::vector<cv::Mat>& descriptors,int current_level);
	virtual void initiateClusters(const std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters) const;
	void initiateClustersKMpp(const std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters) const;
	
	void createWords();
	void setNodeWeights(const std::vector<std::vector<cv::Mat>>& features);	
	void load_fromtxt(const std::string& file_name);

	friend std::ostream& operator<<(std::ostream& os,const Vocabulary& voc);

protected:
	// Branching factor
  	int m_k;
	
	// Depth levels 
  	int m_L;
  
  	// Weighting method
  	WeightingType m_weighting;
	
	// Scoring method
  	ScoringType m_scoring;

	// Object for computing scores
  	GeneralScoring* m_scoring_object;
	
	// Tree nodes
  	std::vector<Node> m_nodes;
	
	std::vector<Node*> m_words;
};
}	// namespace dbow3 

#endif	// VOCABULARY_H_