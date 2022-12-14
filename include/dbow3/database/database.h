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
    explicit Database(bool use_direct_index = true,int levels = 0);
    explicit Database(const Vocabulary& vocabulary,bool use_direct_index = true,int levels = 0);
    Database(const Database& database);
    Database(const std::string& file_name);
    Database(const char* file_name);

    virtual ~Database();
    Database& operator=(const Database& database);

    // set
    void set_vocabulary(const Vocabulary& vocabulary);
    void set_vocabulary(const Vocabulary& vocabulary,bool use_direct_index,int levels = 0);

    // add
    unsigned int add(const cv::Mat& features,BowVector* bowvec = NULL,FeatureVector* fvec = NULL);
    unsigned int add(const std::vector<cv::Mat>& features,BowVector* bowvec = NULL,FeatureVector* fvec = NULL);
    unsigned int add(const BowVector& vec,const FeatureVector& fec = FeatureVector());

    // utils
    const Vocabulary* get_vocabulary() const;
    void allocate(int nd = 0,int ni = 0);
    void clear_database();
    unsigned int entries_size() const { return entries_; }
    bool using_direct_index() const { return use_direct_index_; }
    int get_direct_index_levels() const { return levels_; }
    const FeatureVector& retrieve_features(unsigned int id) const;

    // query
    void query(const cv::Mat& features,QueryResults& ret,int max_results = 1,int max_id = -1) const;
    void query(const std::vector<cv::Mat>& features,QueryResults& ret,int max_results = 1,int max_id = -1) const;
    void query(const BowVector& vec,QueryResults& ret,int max_results = 1,int max_id = -1) const;

    // save
    void save(const std::string& file_name) const;
    virtual void save(cv::FileStorage& fs,const std::string& name = "database") const;

    // load
    void load(const std::string& file_name);
    virtual void load(const cv::FileStorage& fs,const std::string& name = "database");

    // for debug
    void get_info();

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
    Vocabulary* vocabulary_;                        // Associated vocabulary
    std::vector<std::list<IFItem>> inverted_file_;  // Inverted file
    std::vector<FeatureVector> direct_file_;        // Direct file
    bool use_direct_index_;                         // Flag to use direct index
    int levels_;                                    // Levels to go up the vocabulary tree to select nodes to store in the direct index
    int entries_;                                   // Number of valid entries in m_dfile
};
}   // namespace dbow3

#endif  // DATABASE_H_
