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
#include "dbow3/vocabulary/node.h"

namespace dbow3
{
class Vocabulary
{
public:
    Vocabulary(int branching_factor = 10,int depth_levels = 5,WeightingType weighting_type = TF_IDF,ScoringType scoring_type = L1_NORM);
    Vocabulary(const std::string& file_name);
    Vocabulary(const char* file_name);
    Vocabulary(std::istream& stream);
    Vocabulary(const Vocabulary& vocabulary);

    virtual ~Vocabulary();
    Vocabulary& operator=(const Vocabulary& vocabulary);

    // create
    virtual void create(const std::vector<cv::Mat>& training_features);
    virtual void create(const std::vector<std::vector<cv::Mat>>& training_features);
    virtual void create(const std::vector<std::vector<cv::Mat>>& training_features,int branching_factor,int depth_levels);
    virtual void create(const std::vector<std::vector<cv::Mat>>& training_features,int branching_factor,int depth_levels,WeightingType weighting_type,ScoringType scoring_type);

    // utils
    virtual inline unsigned int get_words_size() const;
    virtual inline bool is_empty() const;
    void clear_vocabulary();

    // transform
    virtual void transform(const std::vector<cv::Mat>& features,BowVector& v) const;
    virtual void transform(const cv::Mat& features,BowVector& v) const;
    virtual void transform(const std::vector<cv::Mat>& features,BowVector& v,FeatureVector& fv,int levelsup) const;
    virtual unsigned int transform(const cv::Mat& feature) const;

    // score
    double score(const BowVector& a,const BowVector& b) const;

    // get
    void get_words_from_node(unsigned int node_id,std::vector<unsigned int>& words) const;
    virtual inline cv::Mat get_word(unsigned int word_id) const;
    WeightingType get_weighting_type();
    ScoringType get_scoring_type();
    virtual unsigned int get_parent_node(unsigned int word_id, int levelsup) const;
    int get_descritor_size() const;
    int get_descritor_type() const;
    inline int get_branching_factor() const;
    inline int get_depth_levels() const;
    float get_effective_levels() const;
    virtual inline double get_word_weight(unsigned int word_id) const;

    // set
    inline void set_weighting_type(WeightingType weighting_type);
    void set_scoring_type(ScoringType scoring_type);

    // save
    void save(const std::string& file_name,bool binary_compressed = true) const;
    virtual void save(cv::FileStorage& fs,const std::string& name = "vocabulary") const;

    // load
    void load(const std::string& file_name);
    bool load(std::istream& stream);
    virtual void load(const cv::FileStorage& fs,const std::string& name = "vocabulary");

    // stop
    virtual int stop_words(double weight_min);

    // stream utils
    void to_stream(std::ostream& str,bool compressed=true) const;
    void from_stream(std::istream& str);

    // for debug
    void get_info();

protected:
    void create_scoring_object();
    void get_features(const std::vector<std::vector<cv::Mat>>& training_features,std::vector<cv::Mat>& features) const;

    // transform
    virtual void transform(const cv::Mat& feature,unsigned int& id,double& weight,unsigned int* nid,int levelsup = 0) const;
    virtual void transform(const cv::Mat& feature,unsigned int& id,double& weight ) const;
    virtual void transform(const cv::Mat& feature,unsigned int& id) const;

    // clustering
    void HK_means_step(unsigned int parent_id,const std::vector<cv::Mat>& descriptors,int current_level);
    virtual void initiate_clusters(const std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters) const;
    void initiate_clusters_KMpp(const std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters) const;

    void create_words();
    void set_node_weights(const std::vector<std::vector<cv::Mat>>& features);
    void load_from_txt(const std::string& file_name);

protected:
    int branching_factor_;              // Branching factor
    int depth_levels_;                  // Depth levels
    WeightingType weighting_type_;      // Weighting method
    ScoringType scoring_type_;          // Scoring method
    GeneralScoring* scoring_object_;    // Object for computing scores
    std::vector<Node> nodes_;           // Tree nodes
    std::vector<Node*> words_;          // Words of the vocabulary (tree leaves)
};
}   // namespace dbow3

#endif  // VOCABULARY_H_
