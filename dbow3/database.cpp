#include "dbow3/database/database.h"

using namespace dbow3;

Database::Database(bool use_direct_index,int levels) :
    vocabulary_(NULL),
    use_direct_index_(use_direct_index), levels_(levels), entries_(0) {}

Database::Database(const Vocabulary& vocabulary,bool use_direct_index,int levels) :
    vocabulary_(NULL),
    use_direct_index_(use_direct_index), levels_(levels)
{
    set_vocabulary(vocabulary);
    clear_database();
}

Database::Database(const Database& database) :
    vocabulary_(NULL)
{
    *this = database;
}

Database::Database(const std::string& file_name) :
    vocabulary_(NULL)
{
    load(file_name);
}

Database::Database(const char* file_name) :
    vocabulary_(NULL)
{
    load(file_name);
}

Database::~Database()
{
    delete vocabulary_;
}

Database& Database::operator=(const Database& database)
{
    if(this != &database){
        direct_file_ = database.direct_file_;
        levels_ = database.levels_;
        inverted_file_ = database.inverted_file_;
        entries_ = database.entries_;
        use_direct_index_ = database.use_direct_index_;
        if(database.vocabulary_ != 0) set_vocabulary(*database.vocabulary_);
    }
    return *this;
}

void Database::set_vocabulary(const Vocabulary& vocabulary)
{
    delete vocabulary_;
    vocabulary_ = new Vocabulary(vocabulary);
    clear_database();
}

void Database::set_vocabulary(const Vocabulary& vocabulary,
                              bool use_direct_index,int levels)
{
    use_direct_index_ = use_direct_index;
    levels_ = levels;
    delete vocabulary_;
    vocabulary_ = new Vocabulary(vocabulary);
    clear_database();
}

unsigned int Database::add(const cv::Mat& features,BowVector* bowvec,FeatureVector* fvec)
{
    std::vector<cv::Mat> vf(features.rows);
    for(int r = 0; r < features.rows; r++) vf[r] = features.rowRange(r,r+1);
    return add(vf,bowvec,fvec);
}

unsigned int Database::add(const std::vector<cv::Mat>& features,BowVector* bowvec,FeatureVector* fvec)
{
    BowVector aux;
    BowVector& v = (bowvec ? *bowvec : aux);

    if(use_direct_index_ && fvec != NULL){
        vocabulary_->transform(features,v,*fvec,levels_);
        return add(v,*fvec);
    }
    else if(use_direct_index_){
        FeatureVector fv;
        vocabulary_->transform(features,v,fv,levels_);
        return add(v, fv);
    }
    else if(fvec != NULL){
        vocabulary_->transform(features,v,*fvec,levels_);
        return add(v);
    }
    else{
        vocabulary_->transform(features,v);
        return add(v);
    }
}

unsigned int Database::add(const BowVector& v,const FeatureVector& fv)
{
    unsigned int entry_id = entries_++;

    BowVector::const_iterator vit;
    std::vector<unsigned int>::const_iterator iit;

    if(use_direct_index_){
        // update direct file
        if(entry_id == direct_file_.size()) direct_file_.emplace_back(fv);
        else direct_file_[entry_id] = fv;
    }

    // update inverted file
    for(vit = v.begin(); vit != v.end(); vit++){
        const unsigned int& word_id = vit->first;
        const double& word_weight = vit->second;

        std::list<IFItem>& ifrow = inverted_file_[word_id];
        ifrow.push_back(IFItem(entry_id,word_weight));
    }

    return entry_id;
}

const Vocabulary* Database::get_vocabulary() const
{
    return vocabulary_;
}

void Database::allocate(int nd,int ni)
{
    if(ni > 0){
        for(auto rit = inverted_file_.begin(); rit != inverted_file_.end(); rit++){
            int n = (int)rit->size();
            if(ni > n){
                rit->resize(ni);
                rit->resize(n);
            }
        }
    }

    if(use_direct_index_ && (int)direct_file_.size() < nd) direct_file_.resize(nd);
}

void Database::clear_database()
{
    // resize vectors
    inverted_file_.resize(0);
    inverted_file_.resize(vocabulary_->get_words_size());
    direct_file_.resize(0);
    entries_ = 0;
}

const FeatureVector& Database::retrieve_features(unsigned int id) const
{
    assert(id < entries_size());
    return direct_file_[id];
}

void Database::query(const cv::Mat& features,QueryResults& ret,int max_results,int max_id) const
{
    std::vector<cv::Mat> vf(features.rows);
    for(int r = 0; r < features.rows; r++) vf[r] = features.rowRange(r,r+1);
    query(vf,ret,max_results,max_id);
}

void Database::query(const std::vector<cv::Mat>& features,QueryResults& ret,int max_results,int max_id) const
{
    BowVector vec;
    vocabulary_->transform(features,vec);
    query(vec,ret,max_results,max_id);
}

void Database::query(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const
{
    ret.resize(0);
    switch(vocabulary_->get_scoring_type())
    {
        case L1_NORM:
            queryL1(vec,ret,max_results,max_id);
            break;

        case L2_NORM:
            queryL2(vec,ret,max_results,max_id);
            break;

        case CHI_SQUARE:
            queryChiSquare(vec,ret,max_results,max_id);
            break;

        case KL:
            queryKL(vec,ret,max_results,max_id);
            break;

        case BHATTACHARYYA:
            queryBhattacharyya(vec,ret,max_results,max_id);
            break;

        case DOT_PRODUCT:
            queryDotProduct(vec,ret,max_results,max_id);
            break;
    }
}

void Database::save(const std::string& file_name) const
{
    cv::FileStorage fs(file_name.c_str(),cv::FileStorage::WRITE);
    if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
    save(fs);
}

void Database::save(cv::FileStorage& fs,const std::string& name) const
{
    // database
    vocabulary_->save(fs);
    fs << name << "{";
    fs << "nEntries" << entries_;
    fs << "usingDI" << (use_direct_index_ ? 1 : 0);
    fs << "diLevels" << levels_;

    // inverted_index
    fs << "invertedIndex" << "[";
    for(auto iit = inverted_file_.begin(); iit != inverted_file_.end(); iit++){
        fs << "[";  // word of IF
        for(auto irit = iit->begin(); irit != iit->end(); irit++){
            fs << "{:"
               << "imageId" << (int)irit->entry_id
               << "weight" << irit->word_weight
               << "}";
        }
        fs << "]";
    }
    fs << "]";

    // direct_index
    fs << "directIndex" << "[";
    for(auto dit = direct_file_.begin(); dit != direct_file_.end(); dit++){
        // entry of DF
        fs << "[";
        for(auto drit = dit->begin(); drit != dit->end(); drit++){
            unsigned int nid = drit->first;
            const std::vector<unsigned int>& features = drit->second;

            // save info of last_nid
            fs << "{";
            fs << "nodeId" << (int)nid;
            fs << "features" << "["
               << *(const std::vector<int>*)(&features) << "]";
            fs << "}";
        }
        fs << "]";
    }
    fs << "]";
    fs << "}";
}

void Database::load(const std::string& file_name)
{
    cv::FileStorage fs(file_name.c_str(),cv::FileStorage::READ);
    if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
    load(fs);
}

void Database::load(const cv::FileStorage& fs,const std::string& name)
{
    if(!vocabulary_) vocabulary_ = new Vocabulary;
    vocabulary_->load(fs);

    // load database now
    clear_database();   // resizes inverted file

    cv::FileNode fdb = fs[name];
    entries_ = (int)fdb["nEntries"];
    use_direct_index_ = (int)fdb["usingDI"] != 0;
    levels_ = (int)fdb["diLevels"];

    cv::FileNode fn = fdb["invertedIndex"];
    for(unsigned int wid = 0; wid < fn.size(); wid++){
        cv::FileNode fw = fn[wid];
        for(unsigned int i = 0; i < fw.size(); i++){
            unsigned int eid = (int)fw[i]["imageId"];
            double v = fw[i]["weight"];

            inverted_file_[wid].emplace_back(IFItem(eid,v));
        }
    }

    if(use_direct_index_){
        fn = fdb["directIndex"];

        direct_file_.resize(fn.size());
        assert(entries_ == (int)fn.size());

        FeatureVector::iterator dit;
        for(unsigned int eid = 0; eid < fn.size(); eid++){
            cv::FileNode fe = fn[eid];

            direct_file_[eid].clear();
            for(unsigned int i = 0; i < fe.size(); i++){
                unsigned int nid = (int)fe[i]["nodeId"];
                dit = direct_file_[eid].insert(direct_file_[eid].end(),std::make_pair(nid,std::vector<unsigned int>()));
                cv::FileNode ff = fe[i]["features"][0];
                dit->second.reserve(ff.size());

                cv::FileNodeIterator ffit;
                for(ffit = ff.begin(); ffit != ff.end(); ffit++){
                    dit->second.emplace_back((int)*ffit);
                }
            }
        }
    }
}

void Database::queryL1(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const
{
    std::map<unsigned int,double> pairs;
    std::map<unsigned int,double>::iterator pit;

    BowVector::const_iterator vit;
    for(vit = vec.begin(); vit != vec.end(); vit++){
        const unsigned int word_id = vit->first;
        const double& qvalue = vit->second;

        const std::list<IFItem>& row = inverted_file_[word_id];
        for(auto rit = row.begin(); rit != row.end(); rit++){
            const unsigned int entry_id = rit->entry_id;
            const double& dvalue = rit->word_weight;

            if((int)entry_id < max_id || max_id == -1){
                double value = std::fabs(qvalue - dvalue) - std::fabs(qvalue) - std::fabs(dvalue);

                pit = pairs.lower_bound(entry_id);
                if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
                    pit->second += value;
                }
                else{
                    pairs.insert(pit,std::map<unsigned int,double>::value_type(entry_id,value));
                }
            }
        }
    }

    // move to vector
    ret.reserve(pairs.size());
    for(pit = pairs.begin(); pit != pairs.end(); pit++){
        ret.emplace_back(Result(pit->first, pit->second));
    }
    std::sort(ret.begin(),ret.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);

    QueryResults::iterator qit;
    for(qit = ret.begin(); qit != ret.end(); qit++) qit->score = -qit->score/2.0;
}

void Database::queryL2(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const
{
    BowVector::const_iterator vit;

    std::map<unsigned int,double> pairs;
    std::map<unsigned int,double>::iterator pit;
    for(vit = vec.begin(); vit != vec.end(); vit++){
        const unsigned int word_id = vit->first;
        const double& qvalue = vit->second;

        const std::list<IFItem>& row = inverted_file_[word_id];
        for(auto rit = row.begin(); rit != row.end(); rit++){
            const unsigned int entry_id = rit->entry_id;
            const double& dvalue = rit->word_weight;

            if((int)entry_id < max_id || max_id == -1){
                double value = - qvalue*dvalue; // minus sign for sorting trick

                pit = pairs.lower_bound(entry_id);
                if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
                    pit->second += value;
                }
                else{
                    pairs.insert(pit,std::map<unsigned int,double>::value_type(entry_id,value));
                }
            }
        }
    }

    // move to vector
    ret.reserve(pairs.size());
    for(pit = pairs.begin(); pit != pairs.end(); pit++){
        ret.emplace_back(Result(pit->first, pit->second));
    }
    std::sort(ret.begin(), ret.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);

    QueryResults::iterator qit;
    for(qit = ret.begin(); qit != ret.end(); qit++){
        if(qit->score <= -1.0){
            // rounding error
            qit->score = 1.0;
        }
        else{
            qit->score = 1.0 - std::sqrt(1.0 + qit->score); // [0..1]
        }
    }
}

void Database::queryChiSquare(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const
{
    std::map<unsigned int,std::pair<double,int>> pairs;
    std::map<unsigned int,std::pair<double,int>>::iterator pit;

    std::map<unsigned int,std::pair<double,double>> sums;
    std::map<unsigned int,std::pair<double,double>>::iterator sit;

    BowVector::const_iterator vit;
    for(vit = vec.begin(); vit != vec.end(); vit++){
        const unsigned int word_id = vit->first;
        const double& qvalue = vit->second;

        const std::list<IFItem>& row = inverted_file_[word_id];
        for(auto rit = row.begin(); rit != row.end(); rit++){
            const unsigned int entry_id = rit->entry_id;
            const double& dvalue = rit->word_weight;

            if((int)entry_id < max_id || max_id == -1){
                double value = 0;

                // words may have weight zero
                if(qvalue + dvalue != 0.0) value = - qvalue*dvalue/(qvalue + dvalue);

                pit = pairs.lower_bound(entry_id);
                sit = sums.lower_bound(entry_id);

                if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
                    pit->second.first += value;
                    pit->second.second += 1;
                    sit->second.first += qvalue;
                    sit->second.second += dvalue;
                }
                else{
                    pairs.insert(pit,std::map<unsigned int,std::pair<double,int> >::value_type(entry_id,std::make_pair(value,1)));
                    sums.insert(sit,std::map<unsigned int,std::pair<double,double>>::value_type(entry_id,std::make_pair(qvalue,dvalue)));
                }
            }
        }
    }

    // move to vector
    ret.reserve(pairs.size());
    sit = sums.begin();
    for(pit = pairs.begin(); pit != pairs.end(); pit++, sit++){
        if(pit->second.second >= MIN_COMMON_WORDS){
            ret.emplace_back(Result(pit->first, pit->second.first));
            ret.back().words = pit->second.second;
            ret.back().sum_common_vi = sit->second.first;
            ret.back().sum_common_wi = sit->second.second;
            ret.back().expected_chi_score = 2*sit->second.second/(1 + sit->second.second);
        }
    }
    std::sort(ret.begin(),ret.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);

    // complete and scale score to [0 worst .. 1 best]
    QueryResults::iterator qit;
    for(qit = ret.begin(); qit != ret.end(); qit++){
        // this takes the 4 into account
        qit->score = -2.*qit->score;    // [0..1]
        qit->chi_score = qit->score;
    }
}

void Database::queryKL(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const
{
    std::map<unsigned int,double> pairs;
    std::map<unsigned int,double>::iterator pit;

    BowVector::const_iterator vit;
    for(vit = vec.begin(); vit != vec.end(); vit++){
        const unsigned int word_id = vit->first;
        const double& vi = vit->second;

        const std::list<IFItem>& row = inverted_file_[word_id];
        for(auto rit = row.begin(); rit != row.end(); rit++){
            const unsigned int entry_id = rit->entry_id;
            const double& wi = rit->word_weight;

            if((int)entry_id < max_id || max_id == -1){
                double value = 0;
                if(vi != 0 && wi != 0) value = vi*std::log(vi/wi);

                pit = pairs.lower_bound(entry_id);
                if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
                    pit->second += value;
                }
                else{
                    pairs.insert(pit,std::map<unsigned int,double>::value_type(entry_id,value));
                }
            }
        }
    }

    // complete scores and move to vector
    ret.reserve(pairs.size());
    for(pit = pairs.begin(); pit != pairs.end(); pit++){
        unsigned int eid = pit->first;
        double value = 0.0;
        for(vit = vec.begin(); vit != vec.end(); vit++){
            const double& vi = vit->second;
            const std::list<IFItem>& row = inverted_file_[vit->first];

            if(vi != 0){
                if(row.end() == std::find(row.begin(), row.end(),eid)){
                    value += vi*(std::log(vi) - GeneralScoring::LOG_EPS);
                }
            }
        }
        pit->second += value;

        // to vector
        ret.emplace_back(Result(pit->first, pit->second));
    }
    std::sort(ret.begin(), ret.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
}

void Database::queryBhattacharyya(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const
{
    std::map<unsigned int,std::pair<double,int>> pairs;
    std::map<unsigned int,std::pair<double,int>>::iterator pit;

    BowVector::const_iterator vit;
    for(vit = vec.begin(); vit != vec.end(); vit++){
        const unsigned int word_id = vit->first;
        const double& qvalue = vit->second;

        const std::list<IFItem>& row = inverted_file_[word_id];
        for(auto rit = row.begin(); rit != row.end(); rit++){
            const unsigned int entry_id = rit->entry_id;
            const double& dvalue = rit->word_weight;

            if((int)entry_id < max_id || max_id == -1){
                double value = sqrt(qvalue * dvalue);

                pit = pairs.lower_bound(entry_id);
                if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
                    pit->second.first += value;
                    pit->second.second += 1;
                }
                else{
                    pairs.insert(pit,std::map<unsigned int,std::pair<double,int>>::value_type(entry_id,std::make_pair(value,1)));
                }
            }
        }
    }

    // move to vector
    ret.reserve(pairs.size());
    for(pit = pairs.begin(); pit != pairs.end(); pit++){
        if(pit->second.second >= MIN_COMMON_WORDS){
            ret.emplace_back(Result(pit->first, pit->second.first));
            ret.back().words = pit->second.second;
            ret.back().bhat_score = pit->second.first;
        }
    }
    std::sort(ret.begin(),ret.end(),Result::gt);

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
}

void Database::queryDotProduct(const BowVector& vec,QueryResults& ret,int max_results,int max_id) const
{
    std::map<unsigned int,double> pairs;
    std::map<unsigned int,double>::iterator pit;

    BowVector::const_iterator vit;
    for(vit = vec.begin(); vit != vec.end(); vit++){
        const unsigned int word_id = vit->first;
        const double& qvalue = vit->second;

        const std::list<IFItem>& row = inverted_file_[word_id];
        for(auto rit = row.begin(); rit != row.end(); rit++){
            const unsigned int entry_id = rit->entry_id;
            const double& dvalue = rit->word_weight;

            if((int)entry_id < max_id || max_id == -1){
                double value;
                if(vocabulary_->get_weighting_type() == BINARY) value = 1;
                else value = qvalue * dvalue;

                pit = pairs.lower_bound(entry_id);
                if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
                    pit->second += value;
                }
                else{
                    pairs.insert(pit,std::map<unsigned int,double>::value_type(entry_id,value));
                }
            }
        }
    }

    // move to vector
    ret.reserve(pairs.size());
    for(pit = pairs.begin(); pit != pairs.end(); pit++){
        ret.emplace_back(Result(pit->first, pit->second));
    }
    std::sort(ret.begin(),ret.end(),Result::gt);

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
}

// for debug
void Database::get_info()
{
    std::cout << "Database: Entries = " << entries_size() << std::endl;

    if(using_direct_index()){
        std::cout << "Using direct index = yes" << std::endl;
        std::cout << "Direct index levels = " << get_direct_index_levels();
    }
    else std::cout << "Using direct index = no" << std::endl;
    std::cout << std::endl;

    vocabulary_->get_info();
}
