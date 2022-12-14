#include "dbow3/vocabulary/vocabulary.h"

using namespace dbow3;

Vocabulary::Vocabulary(int k,int L,WeightingType weighting_type,ScoringType scoring_type) :
    branching_factor_(k), depth_levels_(L),
    weighting_type_(weighting_type), scoring_type_(scoring_type),
    scoring_object_(NULL)
{
    create_scoring_object();
}

Vocabulary::Vocabulary(const std::string& file_name) :
    scoring_object_(NULL)
{
    load(file_name);
}

Vocabulary::Vocabulary(const char* file_name) :
    scoring_object_(NULL)
{
    load(file_name);
}

Vocabulary::Vocabulary(std::istream& stream) :
    scoring_object_(NULL)
{
    load(stream);
}

Vocabulary::Vocabulary(const Vocabulary& vocabulary) :
    scoring_object_(NULL)
{
    *this = vocabulary;
}

Vocabulary::~Vocabulary()
{
    delete scoring_object_;
}

Vocabulary& Vocabulary::operator=(const Vocabulary& vocabulary)
{
    branching_factor_ = vocabulary.branching_factor_;
    depth_levels_ = vocabulary.depth_levels_;
    scoring_type_ = vocabulary.scoring_type_;
    weighting_type_ = vocabulary.weighting_type_;
    create_scoring_object();

    nodes_.clear();
    words_.clear();
    nodes_ = vocabulary.nodes_;
    create_words();

    return *this;
}

void Vocabulary::create(const std::vector<cv::Mat>& training_features)
{
    std::vector<std::vector<cv::Mat> > vtf(training_features.size());
    for(size_t i = 0; i < training_features.size(); i++){
        vtf[i].resize(training_features[i].rows);
        for(int r = 0; r < training_features[i].rows; r++){
            vtf[i][r]=training_features[i].rowRange(r,r+1);
        }
    }
    create(vtf);
}

void Vocabulary::create(const std::vector<std::vector<cv::Mat>>& training_features)
{
    nodes_.clear();
    words_.clear();

    int expected_nodes = (int)((std::pow((double)branching_factor_,(double)depth_levels_ + 1) - 1)/(branching_factor_ - 1));
    nodes_.reserve(expected_nodes);

    std::vector<cv::Mat> features;
    get_features(training_features, features);

    // create root
    nodes_.emplace_back(Node(0));   // root

    // create the tree
    HK_means_step(0,features,1);

    // create the words
    create_words();

    // and set the weight of each node of the tree
    set_node_weights(training_features);
}

void Vocabulary::create(const std::vector<std::vector<cv::Mat>>& training_features,int branching_factor,int depth_levels)
{
    branching_factor_ = branching_factor;
    depth_levels_ = depth_levels;
    create(training_features);
}

void Vocabulary::create(const std::vector<std::vector<cv::Mat>>& training_features,int branching_factor,int depth_levels,WeightingType weighting_type,ScoringType scoring_type)
{
    branching_factor_ = branching_factor;
    depth_levels_ = depth_levels;
    weighting_type_ = weighting_type;
    scoring_type_ = scoring_type;
    create_scoring_object();

    create(training_features);
}

inline unsigned int Vocabulary::get_words_size() const
{
    return (unsigned int)words_.size();
}

inline bool Vocabulary::is_empty() const
{
    return words_.empty();
}

void Vocabulary::clear_vocabulary()
{
    delete scoring_object_;
    scoring_object_ = 0;
    nodes_.clear();
    words_.clear();
}

void Vocabulary::transform(const std::vector<cv::Mat>& features,BowVector& v) const
{
    v.clear();
    if(is_empty()) return;

    // normalize
    LNorm norm;
    bool must = scoring_object_->mustNormalize(norm);

    if(weighting_type_ == TF || weighting_type_ == TF_IDF){
        for(auto fit = features.begin(); fit < features.end(); fit++){
            unsigned int id;
            double w;
            transform(*fit,id,w);

            // not stopped
            if(w > 0) v.add_weight(id,w);
        }

        if(!v.empty() && !must){
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
        }
    }
    else{
        for(auto fit = features.begin(); fit < features.end(); fit++){
            unsigned int id;
            double w;
            transform(*fit,id,w);

            // not stopped
            if(w > 0) v.add_if_not_exist(id,w);
        }
    }

    if(must) v.normalize(norm);
}

void Vocabulary::transform(const cv::Mat& features,BowVector& v) const
{
    v.clear();
    if(is_empty()) return;

    // normalize
    LNorm norm;
    bool must = scoring_object_->mustNormalize(norm);

    if(weighting_type_ == TF || weighting_type_ == TF_IDF){
        for(int r = 0; r < features.rows; r++){
            unsigned int id;
            double w;
            transform(features.row(r),id,w);

            // not stopped
            if(w > 0) v.add_weight(id,w);
        }

        if(!v.empty() && !must){
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
        }
    }
    else{
        for(int r = 0; r <features.rows; r++){
            unsigned int id;
            double w;

            transform(features.row(r),id,w);

            // not stopped
            if(w > 0) v.add_if_not_exist(id,w);
        }
    }
    if(must) v.normalize(norm);
}

void Vocabulary::transform(const std::vector<cv::Mat>& features,BowVector& v,FeatureVector& fv,int levelsup) const
{
    v.clear();
    fv.clear();

    if(is_empty()) return;

    // normalize
    LNorm norm;
    bool must = scoring_object_->mustNormalize(norm);

    if(weighting_type_ == TF || weighting_type_ == TF_IDF){
        unsigned int i_feature = 0;
        for(auto fit = features.begin(); fit < features.end(); fit++, i_feature++){
            unsigned int id;
            unsigned int nid;
            double w;
            transform(*fit,id,w,&nid,levelsup);

            // not stopped
            if(w > 0) {
                v.add_weight(id,w);
                fv.add_feature(nid,i_feature);
            }
        }

        if(!v.empty() && !must){
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
        }
    }
    else{
        unsigned int i_feature = 0;
        for(auto fit = features.begin(); fit < features.end(); fit++, i_feature++){
            unsigned int id;
            unsigned int nid;
            double w;
            transform(*fit,id,w,&nid,levelsup);

            // not stopped
            if(w > 0){
                v.add_if_not_exist(id,w);
                fv.add_feature(nid,i_feature);
            }
        }
    }

    if(must) v.normalize(norm);
}

unsigned int Vocabulary::transform(const cv::Mat& feature) const
{
    if(is_empty()) return 0;

    unsigned int wid;
    transform(feature,wid);
    return wid;
}

double Vocabulary::score(const BowVector& a,const BowVector& b) const
{
    return scoring_object_->score(a,b);
}

void Vocabulary::get_words_from_node(unsigned int node_id,std::vector<unsigned int>& words) const
{
    words.clear();
    if(nodes_[node_id].is_leaf()) words.emplace_back(nodes_[node_id].word_id);
    else{
        words.reserve(branching_factor_);
        std::vector<unsigned int> parents;
        parents.emplace_back(node_id);

        while(!parents.empty()){
            unsigned int parentid = parents.back();
            parents.pop_back();

            const std::vector<unsigned int>& child_ids = nodes_[parentid].children;
            std::vector<unsigned int>::const_iterator cit;
            for(cit = child_ids.begin(); cit != child_ids.end(); cit++){
                const Node &child_node = nodes_[*cit];

                if(child_node.is_leaf()) words.emplace_back(child_node.word_id);
                else parents.emplace_back(*cit);
            }
        }
    }
}

cv::Mat Vocabulary::get_word(unsigned int word_id) const
{
    return words_[word_id]->descriptor;
}

WeightingType Vocabulary::get_weighting_type()
{
    return weighting_type_;
}

ScoringType Vocabulary::get_scoring_type()
{
    return scoring_type_;
}

unsigned int Vocabulary::get_parent_node(unsigned int word_id,int levelsup) const
{
    unsigned int ret = words_[word_id]->id;	// node id
    while(levelsup > 0 && ret != 0){
        --levelsup;
        ret = nodes_[ret].parent;
    }
    return ret;
}

int Vocabulary::get_descritor_size() const
{
    if(words_.size() == 0) return -1;
    else return words_[0]->descriptor.cols;
}

int Vocabulary::get_descritor_type() const
{
    if(words_.size() == 0) return -1;
    else return words_[0]->descriptor.type();
}

inline int Vocabulary::get_branching_factor() const
{
    return branching_factor_;
}

inline int Vocabulary::get_depth_levels() const
{
    return depth_levels_;
}

float Vocabulary::get_effective_levels() const
{
    long sum = 0;
    for(auto wit = words_.begin(); wit != words_.end(); wit++){
        const Node* p = *wit;
        for(; p->id != 0; sum++) p = &nodes_[p->parent];
    }
    return (float)((double)sum/(double)words_.size());
}

double Vocabulary::get_word_weight(unsigned int word_id) const
{
    return words_[word_id]->weight;
}

void Vocabulary::set_weighting_type(WeightingType weighting_type)
{
    weighting_type_ = weighting_type;
}

void Vocabulary::set_scoring_type(ScoringType scoring_type)
{
    scoring_type_ = scoring_type;
    create_scoring_object();
}

void Vocabulary::save(const std::string& file_name,bool binary_compressed) const
{
    if(file_name.find(".yml") == std::string::npos){
        std::ofstream file_out(file_name,std::ios::binary);
        if(!file_out) throw std::runtime_error("Vocabulary::saveBinary Could not open file : " +file_name+ " for writing");
        to_stream(file_out,binary_compressed);
    }
    else{
        cv::FileStorage fs(file_name.c_str(),cv::FileStorage::WRITE);
        if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
        save(fs);
    }
}

void Vocabulary::save(cv::FileStorage& f,const std::string &name) const
{
    f << name << "{";
    f << "k" << branching_factor_;
    f << "L" << depth_levels_;
    f << "scoringType" << scoring_type_;
    f << "weightingType" << weighting_type_;

    // tree
    f << "nodes" << "[";
    std::vector<unsigned int> parents, children;
    std::vector<unsigned int>::const_iterator pit;
    parents.emplace_back(0);	// root

    while(!parents.empty()){
        unsigned int pid = parents.back();
        parents.pop_back();

        const Node& parent = nodes_[pid];
        children = parent.children;

        for(pit = children.begin(); pit != children.end(); pit++){
            const Node& child = nodes_[*pit];
            std::cout<< nodes_[*pit].id << " ";

            // save node data
            f << "{:";
            f << "nodeId" << (int)child.id;
            f << "parentId" << (int)pid;
            f << "weight" << (double)child.weight;
            f << "descriptor" << DescriptorsManipulator::to_string(child.descriptor);
            f << "}";

            // add to parent list
            if(!child.is_leaf()) parents.emplace_back(*pit);
        }
    }
    std::cout << "\n";
    f << "]";

    // words
    f << "words" << "[";
    for(auto wit = words_.begin(); wit != words_.end(); wit++){
        unsigned int id = wit - words_.begin();
        f << "{:";
        f << "wordId" << (int)id;
        f << "nodeId" << (int)(*wit)->id;
        f << "}";
    }
    f << "]";
    f << "}";
}

void Vocabulary::load(const std::string& file_name)
{
    // check first if it is a binary file
    std::ifstream ifile(file_name,std::ios::binary);
    if(!ifile) throw std::runtime_error("Vocabulary::load Could not open file : " +file_name+ " for reading");
    if(!load(ifile)){
        if(file_name.find(".txt") != std::string::npos){
            load_from_txt(file_name);
        }else{
            cv::FileStorage fs(file_name.c_str(),cv::FileStorage::READ);
            if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
            load(fs);
        }
    }
}

bool Vocabulary::load(std::istream& ifile)
{
    uint64_t sig;   // magic number describing the file
    ifile.read((char*)&sig,sizeof(sig));

    // Check if it is a binary file.
    if(sig != 88877711233) return false;

    ifile.seekg(0,std::ios::beg);
    from_stream(ifile);
    return true;
}

void Vocabulary::load(const cv::FileStorage& fs,const std::string& name)
{
    words_.clear();
    nodes_.clear();

    cv::FileNode fvoc = fs[name];
    branching_factor_ = (int)fvoc["k"];
    depth_levels_ = (int)fvoc["L"];
    scoring_type_ = (ScoringType)((int)fvoc["scoringType"]);
    weighting_type_ = (WeightingType)((int)fvoc["weightingType"]);

    create_scoring_object();

    // nodes
    cv::FileNode fn = fvoc["nodes"];
    nodes_.resize(fn.size() + 1);
    nodes_[0].id = 0;

    for(unsigned int i = 0; i < fn.size(); i++){
        unsigned int nid = (int)fn[i]["nodeId"];
        unsigned int pid = (int)fn[i]["parentId"];
        double weight = (double)fn[i]["weight"];
        std::string d = (std::string)fn[i]["descriptor"];

        nodes_[nid].id = nid;
        nodes_[nid].parent = pid;
        nodes_[nid].weight = weight;
        nodes_[pid].children.push_back(nid);

        DescriptorsManipulator::from_string(nodes_[nid].descriptor,d);
    }

    // words
    fn = fvoc["words"];
    words_.resize(fn.size());
    for(unsigned int i = 0; i < fn.size(); i++){
        unsigned int wid = (int)fn[i]["wordId"];
        unsigned int nid = (int)fn[i]["nodeId"];

        nodes_[nid].word_id = wid;
        words_[wid] = &nodes_[nid];
    }
}

int Vocabulary::stop_words(double weight_min)
{
    int c = 0;
    for(auto wit = words_.begin(); wit != words_.end(); wit++){
        if((*wit)->weight < weight_min){
            ++c;
            (*wit)->weight = 0;
        }
    }
    return c;
}

void Vocabulary::to_stream(std::ostream& out_str, bool compressed) const
{
    uint64_t sig = 88877711233; // magic number describing the file
    out_str.write((char*)&sig,sizeof(sig));
    out_str.write((char*)&compressed,sizeof(compressed));

    uint32_t nnodes = nodes_.size();
    out_str.write((char*)&nnodes,sizeof(nnodes));
    if(nnodes == 0) return;

    // save everything to a stream
    std::stringstream aux_stream;
    aux_stream.write((char*)&branching_factor_,sizeof(branching_factor_));
    aux_stream.write((char*)&depth_levels_,sizeof(depth_levels_));
    aux_stream.write((char*)&scoring_type_,sizeof(scoring_type_));
    aux_stream.write((char*)&weighting_type_,sizeof(weighting_type_));

    // nodes
    std::vector<unsigned int> parents={0};  // root
    while(!parents.empty()){
        unsigned int pid = parents.back();
        parents.pop_back();

        const Node& parent = nodes_[pid];
        for(auto pit : parent.children){
            const Node& child = nodes_[pit];
            aux_stream.write((char*)&child.id,sizeof(child.id));
            aux_stream.write((char*)&pid,sizeof(pid));
            aux_stream.write((char*)&child.weight,sizeof(child.weight));
            DescriptorsManipulator::to_stream(child.descriptor,aux_stream);

            // add to parent list
            if(!child.is_leaf()) parents.emplace_back(pit);
        }
    }

    // words
    uint32_t m_words_size = words_.size();
    aux_stream.write((char*)&m_words_size,sizeof(m_words_size));
    for(auto wit = words_.begin(); wit != words_.end(); wit++){
        unsigned int id = wit - words_.begin();
        aux_stream.write((char*)&id,sizeof(id));
        aux_stream.write((char*)&(*wit)->id,sizeof((*wit)->id));
    }

    // now, decide if compress or not
    if(compressed){
        QlzStateCompress state_compress;
        memset(&state_compress,0,sizeof(QlzStateCompress));

        // Create output buffer
        int chunkSize = 10000;
        std::vector<char> compressed(chunkSize + size_t(400),0);
        std::vector<char> input(chunkSize, 0);
        int64_t total_size= static_cast<int64_t>(aux_stream.tellp());
        uint64_t total_compress_size=0;

        // calculate how many chunks will be written
        uint32_t nChunks = total_size/chunkSize;
        if(total_size%chunkSize != 0) nChunks++;
        out_str.write((char*)&nChunks,sizeof(nChunks));

        // start compressing the chunks
        while(total_size != 0){
            int readSize = chunkSize;
            if(total_size < chunkSize) readSize = total_size;
            aux_stream.read(&input[0],readSize);

            uint64_t compressed_size = qlz_compress(&input[0],&compressed[0],readSize,&state_compress);
            total_size -= readSize;
            out_str.write(&compressed[0],compressed_size);
            total_compress_size += compressed_size;
        }
    }
    else out_str<<aux_stream.rdbuf();
}

void Vocabulary::from_stream(std::istream& str)
{
    words_.clear();
    nodes_.clear();

    uint64_t sig = 0;   // magic number describing the file
    str.read((char*)&sig,sizeof(sig));
    if(sig != 88877711233) throw std::runtime_error("Vocabulary::fromStream  is not of appropriate type");

    bool compressed;
    str.read((char*)&compressed,sizeof(compressed));

    uint32_t nnodes;
    str.read((char*)&nnodes,sizeof(nnodes));
    if(nnodes == 0)return;

    std::stringstream decompressed_stream;
    std::istream *_used_str=0;
    if(compressed){
        QlzStateDecompress state_decompress;
        memset(&state_decompress,0,sizeof(QlzStateDecompress));

        int chunkSize = 10000;
        std::vector<char> decompressed(chunkSize);
        std::vector<char> input(chunkSize+400);

        // read how many chunks are there
        uint32_t nChunks;
        str.read((char*)&nChunks,sizeof(nChunks));
        for(int i = 0; i < nChunks; i++){
            str.read(&input[0],9);
            int c=qlz_size_compressed(&input[0]);
            str.read(&input[9],c - 9);
            size_t d = qlz_decompress(&input[0],&decompressed[0],&state_decompress);
            decompressed_stream.write(&decompressed[0],d);
        }
        _used_str=&decompressed_stream;
    }
    else{
        _used_str=&str;
    }

    _used_str->read((char*)&branching_factor_,sizeof(branching_factor_));
    _used_str->read((char*)&depth_levels_,sizeof(depth_levels_));
    _used_str->read((char*)&scoring_object_,sizeof(scoring_object_));
    _used_str->read((char*)&weighting_type_,sizeof(weighting_type_));

    create_scoring_object();
    nodes_.resize(nnodes);
    nodes_[0].id = 0;

    for(size_t i = 1; i < nodes_.size(); i++){
        unsigned int nid;
        _used_str->read((char*)&nid,sizeof(unsigned int));

        Node& child = nodes_[nid];
        child.id=nid;
        _used_str->read((char*)&child.parent,sizeof(child.parent));
        _used_str->read((char*)&child.weight,sizeof(child.weight));

        DescriptorsManipulator::from_stream(child.descriptor,*_used_str);
        nodes_[child.parent].children.emplace_back(child.id);
    }

    // words
    uint32_t m_words_size;
    _used_str->read((char*)&m_words_size,sizeof(m_words_size));
    words_.resize(m_words_size);
    for(unsigned int i = 0; i < words_.size(); i++){
        unsigned int wid;
        unsigned int nid;
        _used_str->read((char*)&wid,sizeof(wid));
        _used_str->read((char*)&nid,sizeof(nid));
        nodes_[nid].word_id = wid;
        words_[wid] = &nodes_[nid];
    }
}

void Vocabulary::create_scoring_object()
{
    delete scoring_object_;
    scoring_object_ = NULL;
    switch(scoring_type_)
    {
        case L1_NORM:
            scoring_object_ = new L1Scoring;
            break;

        case L2_NORM:
            scoring_object_ = new L2Scoring;
            break;

        case CHI_SQUARE:
            scoring_object_ = new ChiSquareScoring;
            break;

        case KL:
            scoring_object_ = new KLScoring;
            break;

        case BHATTACHARYYA:
            scoring_object_ = new BhattacharyyaScoring;
            break;

        case DOT_PRODUCT:
            scoring_object_ = new DotProductScoring;
            break;
    }
}

void Vocabulary::get_features(const std::vector<std::vector<cv::Mat>>& training_features,std::vector<cv::Mat>& features) const
{
    features.resize(0);
    for(size_t i = 0; i < training_features.size(); i++){
        for(size_t j = 0; j < training_features[i].size(); j++){
            features.emplace_back(training_features[i][j]);
        }
    }
}

void Vocabulary::transform(const cv::Mat& feature,unsigned int& word_id,double& weight,unsigned int* nid,int levelsup) const
{
    // propagate the feature down the tree

    // level at which the node must be stored in nid, if given
    const int nid_level = depth_levels_ - levelsup;
    if(nid_level <= 0 && nid != NULL) *nid = 0; // root

    unsigned int final_id = 0;  // root
    int current_level = 0;

    do{
        ++current_level;
        auto const  &nodes = nodes_[final_id].children;
        double best_d = std::numeric_limits<double>::max();
        for(const auto &id : nodes){
            double d = DescriptorsManipulator::distance(feature,nodes_[id].descriptor);
            if(d < best_d){
                best_d = d;
                final_id = id;
            }
        }
        if(nid != NULL && current_level == nid_level) *nid = final_id;
    }while(!nodes_[final_id].is_leaf());

    // turn node id into word id
    word_id = nodes_[final_id].word_id;
    weight = nodes_[final_id].weight;
}

void Vocabulary::transform(const cv::Mat& feature,unsigned int& word_id,double& weight) const
{
    // propagate the feature down the tree

    unsigned int final_id = 0;  // root
    if (feature.type() == CV_8U){
        do{
            auto const& nodes = nodes_[final_id].children;
            uint64_t best_d = std::numeric_limits<uint64_t>::max();
            int idx=0,bestidx=0;
            for(const auto &id : nodes){
                // compute distance
                uint64_t dist = DescriptorsManipulator::distance_8uc1(feature,nodes_[id].descriptor);
                if(dist < best_d){
                    best_d = dist;
                    final_id = id;
                    bestidx=idx;
                }
                idx++;
            }
        }while(!nodes_[final_id].is_leaf());
    }
    else{
        do{
            auto const& nodes = nodes_[final_id].children;
            uint64_t best_d = std::numeric_limits<uint64_t>::max();
            int idx = 0, bestidx = 0;
            for(const auto &id : nodes){
                // compute distance
                uint64_t dist = DescriptorsManipulator::distance(feature,nodes_[id].descriptor);
                if(dist < best_d){
                    best_d = dist;
                    final_id = id;
                    bestidx = idx;
                }
                idx++;
            }
        }while(!nodes_[final_id].is_leaf());
    }

    // turn node id into word id
    word_id = nodes_[final_id].word_id;
    weight = nodes_[final_id].weight;
}

void Vocabulary::transform(const cv::Mat& feature,unsigned int& id) const
{
    double weight;
    transform(feature,id,weight);
}

void Vocabulary::HK_means_step(unsigned int parent_id,const std::vector<cv::Mat>& descriptors,int current_level)
{
    if(descriptors.empty()) return;

    // features associated to each cluster
    std::vector<cv::Mat> clusters;
    std::vector<std::vector<unsigned int>> groups;
    clusters.reserve(branching_factor_);
    groups.reserve(branching_factor_);

    if((int)descriptors.size() <= branching_factor_){
        // trivial case: one cluster per feature
        groups.resize(descriptors.size());
        for(unsigned int i = 0; i < descriptors.size(); i++){
            groups[i].emplace_back(i);
            clusters.emplace_back(descriptors[i]);
        }
    }
    else{
        // select clusters and groups with kmeans
        bool first_time = true;
        bool goon = true;

        // to check if clusters move after iterations
        std::vector<int> last_association, current_association;

        while(goon){
            // 1. Calculate clusters
            if(first_time){
                // random sample
                initiate_clusters(descriptors,clusters);
            }
            else{
                // calculate cluster centres
                for(unsigned int c = 0; c < clusters.size(); c++){
                    std::vector<cv::Mat> cluster_descriptors;
                    cluster_descriptors.reserve(groups[c].size());
                    std::vector<unsigned int>::const_iterator vit;
                    for(vit = groups[c].begin(); vit != groups[c].end(); vit++){
                        cluster_descriptors.emplace_back(descriptors[*vit]);
                    }
                    DescriptorsManipulator::mean_value(cluster_descriptors,clusters[c]);
                }

            }

            // 2. Associate features with clusters
            // calculate distances to cluster centers
            groups.clear();
            groups.resize(clusters.size(),std::vector<unsigned int>());
            current_association.resize(descriptors.size());
            for(auto fit = descriptors.begin(); fit != descriptors.end(); fit++){
                double best_dist = DescriptorsManipulator::distance((*fit),clusters[0]);
                unsigned int icluster = 0;
                for(unsigned int c = 1; c < clusters.size(); c++){
                    double dist = DescriptorsManipulator::distance((*fit),clusters[c]);
                    if(dist < best_dist){
                        best_dist = dist;
                        icluster = c;
                    }
                }

                groups[icluster].emplace_back(fit - descriptors.begin());
                current_association[fit - descriptors.begin()] = icluster;
            }

            // kmeans++ ensures all the clusters has any feature associated with them
            // 3. check convergence
            if(first_time) first_time = false;
            else
            {
                goon = false;
                for(unsigned int i = 0; i < current_association.size(); i++){
                    if(current_association[i] != last_association[i]){
                        goon = true;
                        break;
                    }
                }
            }

            if(goon){
                // copy last feature-cluster association
                last_association = current_association;
            }
        }
    }

    // create nodes
    for(unsigned int i = 0; i < clusters.size(); i++){
        unsigned int id = nodes_.size();
        nodes_.emplace_back(Node(id));
        nodes_.back().descriptor = clusters[i];
        nodes_.back().parent = parent_id;
        nodes_[parent_id].children.emplace_back(id);
    }

    // go on with the next level
    if(current_level < depth_levels_){
        // iterate again with the resulting clusters
        const std::vector<unsigned int>& children_ids = nodes_[parent_id].children;
        for(unsigned int i = 0; i < clusters.size(); i++){
            unsigned int id = children_ids[i];

            std::vector<cv::Mat> child_features;
            child_features.reserve(groups[i].size());

            std::vector<unsigned int>::const_iterator vit;
            for(vit = groups[i].begin(); vit != groups[i].end(); vit++){
                child_features.emplace_back(descriptors[*vit]);
            }

            if(child_features.size() > 1) HK_means_step(id,child_features,current_level + 1);
        }
    }
}

void Vocabulary::initiate_clusters(const std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters) const
{
    initiate_clusters_KMpp(descriptors, clusters);
}

void Vocabulary::initiate_clusters_KMpp(const std::vector<cv::Mat>& pfeatures,std::vector<cv::Mat>& clusters) const
{
    clusters.resize(0);
    clusters.reserve(branching_factor_);
    std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());

    // 1. Choose one center uniformly at random from among the data points.
    int ifeature = std::rand()% pfeatures.size();

    // create first cluster
    clusters.push_back(pfeatures[ifeature]);

    // compute the initial distances
    std::vector<double>::iterator dit = min_dists.begin();
    for(auto fit = pfeatures.begin(); fit != pfeatures.end(); fit++, dit++){
        *dit = DescriptorsManipulator::distance((*fit),clusters.back());
    }

    while((int)clusters.size() < branching_factor_){
        // 2. For each data point x, compute D(x),
        //    the distance between x and the nearest center that has already been chosen.
        dit = min_dists.begin();
        for(auto fit = pfeatures.begin(); fit != pfeatures.end(); fit++, dit++){
            if(*dit > 0){
                double dist = DescriptorsManipulator::distance((*fit),clusters.back());
                if(dist < *dit) *dit = dist;
            }
        }
        // 3. Add one new data point as a center.
        //    Each point x is chosen with probability proportional to D(x)^2.
        double dist_sum = std::accumulate(min_dists.begin(),min_dists.end(),0.0);
        if(dist_sum > 0){
            double cut_d;
            do{
                cut_d = (double(std::rand())/double(RAND_MAX))*dist_sum;
            }while(cut_d == 0.0);

            double d_up_now = 0;
            for(dit = min_dists.begin(); dit != min_dists.end(); dit++){
                d_up_now += *dit;
                if(d_up_now >= cut_d) break;
            }

            if(dit == min_dists.end()) ifeature = pfeatures.size()-1;
            else ifeature = dit - min_dists.begin();

            clusters.emplace_back(pfeatures[ifeature]);
        }
        else break;
    }
}

void Vocabulary::create_words()
{
    words_.resize(0);
    if(!nodes_.empty()){
        words_.reserve((int)std::pow((double)branching_factor_,(double)depth_levels_));

        auto nit = nodes_.begin();	// ignore root
        for(++nit; nit != nodes_.end(); nit++){
            if(nit->is_leaf()){
                nit->word_id = words_.size();
                words_.emplace_back( &(*nit) );
            }
        }
    }
}

void Vocabulary::set_node_weights(const std::vector<std::vector<cv::Mat>>& training_features)
{
    const unsigned int NWords = words_.size();
    const unsigned int NDocs = training_features.size();
    if(weighting_type_ == TF || weighting_type_ == BINARY){
        // idf part must be 1 always
        for(unsigned int i = 0; i < NWords; i++) words_[i]->weight = 1;
    }
    else if(weighting_type_ == IDF || weighting_type_ == TF_IDF){
        // IDF and TF-IDF: we calculte the idf path now
        std::vector<unsigned int> Ni(NWords, 0);
        std::vector<bool> counted(NWords, false);
        for(auto mit = training_features.begin(); mit != training_features.end(); mit++){
            std::fill(counted.begin(),counted.end(),false);
            for(auto fit = mit->begin(); fit < mit->end(); fit++){
                unsigned int word_id;
                transform(*fit,word_id);

                if(!counted[word_id]){
                    Ni[word_id]++;
                    counted[word_id] = true;
                }
            }
        }

        // set ln(N/Ni)
        for(unsigned int i = 0; i < NWords; i++){
            if(Ni[i] > 0){
                words_[i]->weight = std::log((double)NDocs/(double)Ni[i]);
            }
        }
    }
}

void Vocabulary::load_from_txt(const std::string& file_name)
{
    std::ifstream ifile(file_name);
    if(!ifile) throw std::runtime_error("Vocabulary:: load_fromtxt  Could not open file for reading: " + file_name);

    int n1, n2;
    {
        std::string str;
        getline(ifile,str);
        std::stringstream ss(str);
        ss >> branching_factor_ >> depth_levels_ >> n1 >> n2;
    }

    if(branching_factor_ < 0 || branching_factor_ > 20 || depth_levels_ < 1 || depth_levels_ > 10 || n1 < 0 || n1 > 5 || n2 < 0 || n2 > 3){
        throw std::runtime_error( "Vocabulary loading failure: This is not a correct text file!" );
    }
    scoring_type_ = (ScoringType)n1;
    weighting_type_ = (WeightingType)n2;
    create_scoring_object();

    // nodes
    int expected_nodes = (int)((std::pow((double)branching_factor_,(double)depth_levels_ + 1) - 1)/(branching_factor_ - 1));
    nodes_.reserve(expected_nodes);
    words_.reserve(std::pow((double)branching_factor_,(double)depth_levels_ + 1));
    nodes_.resize(1);
    nodes_[0].id = 0;

    int counter=0;
    while(!ifile.eof()){
        std::string snode;
        getline(ifile,snode);
        if(counter++ % 100 == 0) std::cerr <<".";
        if(snode.size() == 0) break;

        std::stringstream ssnode(snode);

        int nid = nodes_.size();
        nodes_.resize(nodes_.size() + 1);
        nodes_[nid].id = nid;

        int pid ;
        ssnode >> pid;
        nodes_[nid].parent = pid;
        nodes_[pid].children.emplace_back(nid);

        int nIsLeaf;
        ssnode >> nIsLeaf;

        // read until the end and add to data
        std::vector<float> data;
        data.reserve(100);

        float d;
        while(ssnode >> d) data.emplace_back(d);

        // the weight is the last
        nodes_[nid].weight = data.back();
        data.pop_back();    // remove

        // the rest, to the descriptor
        nodes_[nid].descriptor.create(1,data.size(),CV_8UC1);
        auto ptr = nodes_[nid].descriptor.ptr<uchar>(0);
        for(auto d : data) *ptr++ = d;

        if(nIsLeaf > 0){
            int wid = words_.size();
            words_.resize(wid + 1);

            nodes_[nid].word_id = wid;
            words_[wid] = &nodes_[nid];
        }
        else{
            nodes_[nid].children.reserve(branching_factor_);
        }
    }
}

void Vocabulary::get_info()
{
    std::cout << "k = " << get_branching_factor() << std::endl;
    std::cout << "L = " << get_depth_levels() << std::endl;

    std::cout << "Weighting = ";
    switch(get_weighting_type())
    {
        case TF_IDF:
            std::cout << "tf-idf" << std::endl;
            break;

        case TF:
            std::cout << "tf" << std::endl;
            break;

        case IDF:
            std::cout << "idf" << std::endl;
            break;

        case BINARY:
            std::cout << "binary" << std::endl;
            break;
    }

    std::cout << "Scoring = ";
    switch(get_descritor_type())
    {
        case L1_NORM:
            std::cout << "L1-norm" << std::endl;
            break;

        case L2_NORM:
            std::cout << "L2-norm" << std::endl;
            break;

        case CHI_SQUARE:
            std::cout << "Chi square distance" << std::endl;
            break;

        case KL:
            std::cout << "KL-divergence" << std::endl;
            break;

        case BHATTACHARYYA:
            std::cout << "Bhattacharyya coefficient" << std::endl;
            break;

        case DOT_PRODUCT:
            std::cout << "Dot product" << std::endl;
            break;
    }

    std::cout << "Number of words = " << get_words_size() << std::endl;
    std::cout << std::endl;

}
