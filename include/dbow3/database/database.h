#ifndef DATABASE_H_
#define DATABASE_H_

#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <list>
#include <set>

#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/query_results/query_results.h"
#include "dbow3/scoring_object/scoring_object.h"
#include "dbow3/bow_vector/bow_vector.h"
#include "dbow3/feature_vector/feature_vector.h"
#include "dbow3/database/if_item.h"

namespace dbow3
{
// For query functions
const int MIN_COMMON_WORDS = 5;

class Database
{
public:
	explicit Database(bool use_di = true,int di_levels = 0);
	explicit Database(const Vocabulary& voc,bool use_di = true,int di_levels = 0);
	Database(const Database& db);
	Database(const std::string& file_name);
	Database(const char* file_name);

  	virtual ~Database();
	Database& operator=(const Database& db);

  	void setVocabulary(const Vocabulary& voc);
	void setVocabulary(const Vocabulary& voc,bool use_di,int di_levels = 0);
	
	const Vocabulary* getVocabulary() const;
	void allocate(int nd = 0,int ni = 0);
	
	// add
	unsigned int add(const cv::Mat& features,BowVector* bowvec = NULL,FeatureVector* fvec = NULL);
	unsigned int add(const std::vector<cv::Mat>& features,BowVector* bowvec = NULL,FeatureVector* fvec = NULL);
	unsigned int add(const BowVector& vec,const FeatureVector& fec = FeatureVector());
	
	void clear();
	unsigned int size() const{  return m_nentries;}
	bool usingDirectIndex() const{  return m_use_di;}
	int getDirectIndexLevels() const{  return m_dilevels;}

	void query(const cv::Mat& features,QueryResults& ret,int max_results = 1,int max_id = -1) const;
  	void query(const std::vector<cv::Mat>& features,QueryResults& ret,int max_results = 1,int max_id = -1) const;
	void query(const BowVector& vec,QueryResults& ret,int max_results = 1,int max_id = -1) const;

  	const FeatureVector& retrieveFeatures(unsigned int id) const;

  	void save(const std::string& file_name) const;
	virtual void save(cv::FileStorage& fs,const std::string& name = "database") const;

	void load(const std::string& file_name);
	virtual void load(const cv::FileStorage& fs,const std::string& name = "database");
	
	// for debug
	void get_info();

	// friend std::ostream& operator<<(std::ostream os,const Database& db);

protected:
	// Query with L1 scoring
  	void queryL1(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const;
	
	// Query with L2 scoring
  	void queryL2(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const;
	
	// Query with Chi square scoring
  	void queryChiSquare(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const;
	
	// Query with Bhattacharyya scoring
	void queryBhattacharyya(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const;
	
	// Query with KL divergence scoring  
  	void queryKL(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const;
	
	// Query with dot product scoring
  	void queryDotProduct(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const;
	
protected:
	// Associated vocabulary
  	Vocabulary *m_voc;
	
	// Flag to use direct index
  	bool m_use_di;
	
	// Levels to go up the vocabulary tree to select nodes to store in the direct index
  	int m_dilevels;
	
	// Inverted file (must have size() == |words|)
  	std::vector<std::list<IFItem>> m_ifile;
	
	// Direct file (resized for allocation)
  	std::vector<FeatureVector> m_dfile;
	
	// Number of valid entries in m_dfile
  	int m_nentries;  
};
}	// namespace dbow3

#endif	// DATABASE_H_