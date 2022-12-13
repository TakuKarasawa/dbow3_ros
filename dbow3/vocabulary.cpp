#include "dbow3/vocabulary/vocabulary.h"

using namespace dbow3;

Vocabulary::Vocabulary(int k,int L,WeightingType weighting,ScoringType scoring) : 
	m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
	m_scoring_object(NULL)
{
	createScoringObject();
}

Vocabulary::Vocabulary(const std::string& file_name) : 
	m_scoring_object(NULL)
{
	load(file_name);
}

Vocabulary::Vocabulary(const char* file_name) : 
	m_scoring_object(NULL)
{
	load(file_name);
}

Vocabulary::Vocabulary(std::istream& stream) : 
	m_scoring_object(NULL)
{
	load(stream);
}

Vocabulary::Vocabulary(const Vocabulary& voc) : 
	m_scoring_object(NULL)
{
	*this = voc;
}

Vocabulary::~Vocabulary()
{
	delete m_scoring_object;
}

Vocabulary& Vocabulary::operator=(const Vocabulary &voc)
{
	this->m_k = voc.m_k;
  	this->m_L = voc.m_L;
  	this->m_scoring = voc.m_scoring;
  	this->m_weighting = voc.m_weighting;
	
	this->createScoringObject();
	
	this->m_nodes.clear();
	this->m_words.clear();

  	this->m_nodes = voc.m_nodes;
  	this->createWords();
	
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
	m_nodes.clear();
	m_words.clear();
	
	int expected_nodes = (int)((std::pow((double)m_k,(double)m_L + 1) - 1)/(m_k - 1));
	m_nodes.reserve(expected_nodes); 

  	std::vector<cv::Mat> features;
  	getFeatures(training_features, features);
	
	// create root
	m_nodes.push_back(Node(0));	// root
	
	// create the tree
  	HKmeansStep(0,features,1);
	
	// create the words
  	createWords();
	
	// and set the weight of each node of the tree
  	setNodeWeights(training_features);
}

void Vocabulary::create(const std::vector<std::vector<cv::Mat>>& training_features,int k,int L)
{
	m_k = k;
  	m_L = L;
	create(training_features);
}

void Vocabulary::create(const std::vector<std::vector<cv::Mat>>& training_features,int k,int L,WeightingType weighting,ScoringType scoring)
{
	m_k = k;
  	m_L = L;
  	m_weighting = weighting;
  	m_scoring = scoring;
  	createScoringObject();
	
	create(training_features);
}

inline unsigned int Vocabulary::size() const
{ 
	return (unsigned int)m_words.size(); 
}

inline bool Vocabulary::empty() const
{ 
	return m_words.empty(); 
}

void Vocabulary::clear()
{
	delete m_scoring_object;
    m_scoring_object = 0;
    m_nodes.clear();
    m_words.clear();
}

void Vocabulary::transform(const std::vector<cv::Mat>& features,BowVector& v) const
{
	v.clear();
	if(empty()) return;
	
	// normalize
	LNorm norm;
  	bool must = m_scoring_object->mustNormalize(norm);
  	
	if(m_weighting == TF || m_weighting == TF_IDF){
		for(auto fit = features.begin(); fit < features.end(); fit++){
			WordId id;
      		WordValue w;
			transform(*fit,id,w);
			
			// not stopped
			if(w > 0) v.addWeight(id,w);
    	}
		
		if(!v.empty() && !must){
			// unnecessary when normalizing
      		const double nd = v.size();
      		for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
    	}
	}
  	else{
		for(auto fit = features.begin(); fit < features.end(); fit++){
			WordId id;
      		WordValue w;
			transform(*fit,id,w);
			
			// not stopped
			if(w > 0) v.addIfNotExist(id,w);
		}
  	}
	
	if(must) v.normalize(norm);
}

void Vocabulary::transform(const cv::Mat& features,BowVector& v) const
{
	v.clear();
	if(empty()) return;
    
    // normalize
    LNorm norm;
    bool must = m_scoring_object->mustNormalize(norm);

    if(m_weighting == TF || m_weighting == TF_IDF){
		for(int r = 0; r < features.rows; r++){
			WordId id;
            WordValue w;
            transform(features.row(r),id,w);

            // not stopped
            if(w > 0) v.addWeight(id,w);
        }

        if(!v.empty() && !must){
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
        }
    }
    else{
        for(int r = 0; r <features.rows; r++){
			WordId id;
            WordValue w;

            transform(features.row(r),id,w);

            // not stopped
            if(w > 0) v.addIfNotExist(id,w);
        }
    }
    if(must) v.normalize(norm);
}

void Vocabulary::transform(const std::vector<cv::Mat>& features,BowVector& v,FeatureVector& fv,int levelsup) const
{
	v.clear();
  	fv.clear();
	
	if(empty()) return;
	
	// normalize
  	LNorm norm;
  	bool must = m_scoring_object->mustNormalize(norm);
	
	if(m_weighting == TF || m_weighting == TF_IDF){
		unsigned int i_feature = 0;
    	for(auto fit = features.begin(); fit < features.end(); fit++, i_feature++){
			WordId id;
      		NodeId nid;
      		WordValue w;
      		transform(*fit,id,w,&nid,levelsup);

			// not stopped
      		if(w > 0) {
				v.addWeight(id,w);
				fv.addFeature(nid,i_feature);
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
			WordId id;
      		NodeId nid;
      		WordValue w;
			transform(*fit,id,w,&nid,levelsup);

			// not stopped
      		if(w > 0){
				v.addIfNotExist(id,w);
				fv.addFeature(nid,i_feature);
      		}
    	}
  	} 
	
	if(must) v.normalize(norm);
}

WordId Vocabulary::transform(const cv::Mat& feature) const
{
	if(empty()) return 0;
	
	WordId wid;
	transform(feature,wid);
	return wid;
}

double Vocabulary::score(const BowVector& a,const BowVector& b) const 
{ 
	return m_scoring_object->score(a,b); 
}

NodeId Vocabulary::getParentNode(WordId wid,int levelsup) const
{
	NodeId ret = m_words[wid]->id;	// node id
  	while(levelsup > 0 && ret != 0){
		--levelsup;
    	ret = m_nodes[ret].parent;
  	}
	return ret;
}

void Vocabulary::getWordsFromNode(NodeId nid,std::vector<WordId>& words) const
{
	words.clear();
	if(m_nodes[nid].isLeaf()) words.emplace_back(m_nodes[nid].word_id);
	else{
		words.reserve(m_k);
		std::vector<NodeId> parents;
		parents.emplace_back(nid);

    	while(!parents.empty()){
			NodeId parentid = parents.back();
      		parents.pop_back();

      		const std::vector<NodeId> &child_ids = m_nodes[parentid].children;
      		std::vector<NodeId>::const_iterator cit;
			for(cit = child_ids.begin(); cit != child_ids.end(); cit++){
				const Node &child_node = m_nodes[*cit];
				
				if(child_node.isLeaf()) words.emplace_back(child_node.word_id);
        		else parents.emplace_back(*cit);
      		} 
    	} 
  	}
}

inline int Vocabulary::getBranchingFactor() const 
{ 
	return m_k; 
}

inline int Vocabulary::getDepthLevels() const 
{ 
	return m_L; 
}

float Vocabulary::getEffectiveLevels() const
{
	long sum = 0;
	for(auto wit = m_words.begin(); wit != m_words.end(); wit++){
		const Node* p = *wit;
		for(; p->id != 0; sum++) p = &m_nodes[p->parent];
  	}
	return (float)((double)sum/(double)m_words.size());
}

cv::Mat Vocabulary::getWord(WordId wid) const
{
	return m_words[wid]->descriptor;
}

WordValue Vocabulary::getWordWeight(WordId wid) const
{
	return m_words[wid]->weight;
}

/*
inline WeightingType Vocabulary::getWeightingType()
{ 
	return m_weighting; 
}

inline ScoringType Vocabulary::getScoringType()
{ 
	return m_scoring; 
}
*/

WeightingType Vocabulary::getWeightingType()
{ 
	return m_weighting; 
}

ScoringType Vocabulary::getScoringType()
{ 
	return m_scoring; 
}

void Vocabulary::setScoringType(ScoringType type)
{
	m_scoring = type;
	createScoringObject();
}

void Vocabulary::setWeightingType(WeightingType type)
{
	this->m_weighting = type;
}

void Vocabulary::save(const std::string& file_name,bool binary_compressed) const
{
	if(file_name.find(".yml") == std::string::npos){
		std::ofstream file_out(file_name,std::ios::binary);
        if(!file_out) throw std::runtime_error("Vocabulary::saveBinary Could not open file : " +file_name+ " for writing");
        toStream(file_out,binary_compressed);
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
	f << "k" << m_k;
 	f << "L" << m_L;
  	f << "scoringType" << m_scoring;
  	f << "weightingType" << m_weighting;

  	// tree
  	f << "nodes" << "[";
  	std::vector<NodeId> parents, children;
  	std::vector<NodeId>::const_iterator pit;
  	parents.emplace_back(0);	// root

  	while(!parents.empty()){
		NodeId pid = parents.back();
    	parents.pop_back();

    	const Node& parent = m_nodes[pid];
    	children = parent.children;

    	for(pit = children.begin(); pit != children.end(); pit++){
			const Node& child = m_nodes[*pit];
      		std::cout<<m_nodes[*pit].id << " ";
			
			// save node data
      		f << "{:";
      		f << "nodeId" << (int)child.id;
      		f << "parentId" << (int)pid;
      		f << "weight" << (double)child.weight;
      		f << "descriptor" << DescriptorsManipulator::toString(child.descriptor);
      		f << "}";
			
			// add to parent list
			if(!child.isLeaf()) parents.emplace_back(*pit); 
    	}
  	}
  	std::cout << "\n";
	f << "]";

  	// words
  	f << "words" << "[";
	for(auto wit = m_words.begin(); wit != m_words.end(); wit++){
		WordId id = wit - m_words.begin();
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
			load_fromtxt(file_name);
		}else{
			cv::FileStorage fs(file_name.c_str(),cv::FileStorage::READ);
	    	if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
	    	load(fs);
		}
    }
}

bool Vocabulary::load(std::istream& ifile)
{
    uint64_t sig;	// magic number describing the file
    ifile.read((char*)&sig,sizeof(sig));

    // Check if it is a binary file.
	if(sig != 88877711233) return false;
	
	ifile.seekg(0,std::ios::beg);
    fromStream(ifile);
    return true;
}

void Vocabulary::load(const cv::FileStorage& fs,const std::string& name)
{
	m_words.clear();
	m_nodes.clear();

  	cv::FileNode fvoc = fs[name];
  	m_k = (int)fvoc["k"];
  	m_L = (int)fvoc["L"];
  	m_scoring = (ScoringType)((int)fvoc["scoringType"]);
  	m_weighting = (WeightingType)((int)fvoc["weightingType"]);

  	createScoringObject();
	
	// nodes
  	cv::FileNode fn = fvoc["nodes"];
	m_nodes.resize(fn.size() + 1);
  	m_nodes[0].id = 0;
	
	for(unsigned int i = 0; i < fn.size(); i++){
		NodeId nid = (int)fn[i]["nodeId"];
    	NodeId pid = (int)fn[i]["parentId"];
    	WordValue weight = (WordValue)fn[i]["weight"];
    	std::string d = (std::string)fn[i]["descriptor"];

    	m_nodes[nid].id = nid;
    	m_nodes[nid].parent = pid;
    	m_nodes[nid].weight = weight;
    	m_nodes[pid].children.push_back(nid);

		DescriptorsManipulator::fromString(m_nodes[nid].descriptor,d);
  	}
	
	// words
  	fn = fvoc["words"];
	m_words.resize(fn.size());
	for(unsigned int i = 0; i < fn.size(); i++){
		NodeId wid = (int)fn[i]["wordId"];
    	NodeId nid = (int)fn[i]["nodeId"];

    	m_nodes[nid].word_id = wid;
    	m_words[wid] = &m_nodes[nid];
  	}
}

int Vocabulary::stopWords(double minWeight)
{
	int c = 0;
	for(auto wit = m_words.begin(); wit != m_words.end(); wit++){
		if((*wit)->weight < minWeight){
			++c;
      		(*wit)->weight = 0;
    	}
  	}
  	return c;
}

int Vocabulary::getDescritorSize() const
{
	if(m_words.size() == 0) return -1;
    else return m_words[0]->descriptor.cols;
}

int Vocabulary::getDescritorType() const
{
	if(m_words.size() == 0) return -1;
    else return m_words[0]->descriptor.type();
}

void Vocabulary::toStream(std::ostream& out_str, bool compressed) const
{
	uint64_t sig = 88877711233;	// magic number describing the file
    out_str.write((char*)&sig,sizeof(sig));
    out_str.write((char*)&compressed,sizeof(compressed));
    
	uint32_t nnodes = m_nodes.size();
    out_str.write((char*)&nnodes,sizeof(nnodes));
    if(nnodes == 0) return;

    // save everything to a stream
    std::stringstream aux_stream;
    aux_stream.write((char*)&m_k,sizeof(m_k));
    aux_stream.write((char*)&m_L,sizeof(m_L));
    aux_stream.write((char*)&m_scoring,sizeof(m_scoring));
    aux_stream.write((char*)&m_weighting,sizeof(m_weighting));
    
	// nodes
    std::vector<NodeId> parents={0};	// root
    while(!parents.empty()){
		NodeId pid = parents.back();
        parents.pop_back();
        
		const Node& parent = m_nodes[pid];
        for(auto pit : parent.children){
			const Node& child = m_nodes[pit];
            aux_stream.write((char*)&child.id,sizeof(child.id));
            aux_stream.write((char*)&pid,sizeof(pid));
            aux_stream.write((char*)&child.weight,sizeof(child.weight));
			DescriptorsManipulator::toStream(child.descriptor,aux_stream);
            
			// add to parent list
            if(!child.isLeaf()) parents.emplace_back(pit);
        }
    }

    // words
    uint32_t m_words_size = m_words.size();
    aux_stream.write((char*)&m_words_size,sizeof(m_words_size));
    for(auto wit = m_words.begin(); wit != m_words.end(); wit++){
		WordId id = wit - m_words.begin();
        aux_stream.write((char*)&id,sizeof(id));
        aux_stream.write((char*)&(*wit)->id,sizeof((*wit)->id));
    }

    // now, decide if compress or not
    if(compressed){
		qlz_state_compress  state_compress;
        memset(&state_compress,0,sizeof(qlz_state_compress));
        
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

void Vocabulary::fromStream(std::istream& str)
{
	m_words.clear();
    m_nodes.clear();

    uint64_t sig = 0;	// magic number describing the file
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
		qlz_state_decompress state_decompress;
        memset(&state_decompress,0,sizeof(qlz_state_decompress));
        
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

    _used_str->read((char*)&m_k,sizeof(m_k));
    _used_str->read((char*)&m_L,sizeof(m_L));
    _used_str->read((char*)&m_scoring,sizeof(m_scoring));
    _used_str->read((char*)&m_weighting,sizeof(m_weighting));

    createScoringObject();
    m_nodes.resize(nnodes);
    m_nodes[0].id = 0;

    for(size_t i = 1; i < m_nodes.size(); i++){
		NodeId nid;
        _used_str->read((char*)&nid,sizeof(NodeId));
        
		Node& child = m_nodes[nid];
        child.id=nid;
        _used_str->read((char*)&child.parent,sizeof(child.parent));
        _used_str->read((char*)&child.weight,sizeof(child.weight));
        
		DescriptorsManipulator::fromStream(child.descriptor,*_used_str);
        m_nodes[child.parent].children.emplace_back(child.id);
    }
    
	// words
    uint32_t m_words_size;
    _used_str->read((char*)&m_words_size,sizeof(m_words_size));
    m_words.resize(m_words_size);
    for(unsigned int i = 0; i < m_words.size(); i++){
        WordId wid;NodeId nid;
        _used_str->read((char*)&wid,sizeof(wid));
        _used_str->read((char*)&nid,sizeof(nid));
        m_nodes[nid].word_id = wid;
        m_words[wid] = &m_nodes[nid];
    }
}

void Vocabulary::createScoringObject()
{
	delete m_scoring_object;
  	m_scoring_object = NULL;
	switch(m_scoring)
	{
		case L1_NORM:
			m_scoring_object = new L1Scoring;
      		break;

    	case L2_NORM:
     	 	m_scoring_object = new L2Scoring;
      		break;

    	case CHI_SQUARE:
      		m_scoring_object = new ChiSquareScoring;
      		break;

    	case KL:
      		m_scoring_object = new KLScoring;
      		break;

    	case BHATTACHARYYA:
      		m_scoring_object = new BhattacharyyaScoring;
      		break;

    	case DOT_PRODUCT:
      		m_scoring_object = new DotProductScoring;
      		break;
  	}
}

void Vocabulary::getFeatures(const std::vector<std::vector<cv::Mat>>& training_features,std::vector<cv::Mat>& features) const
{
	features.resize(0);
  	for(size_t i = 0; i < training_features.size(); i++){
    	for(size_t j = 0; j < training_features[i].size(); j++){
			features.emplace_back(training_features[i][j]);
		}
	}
}

void Vocabulary::transform(const cv::Mat& feature,WordId& word_id,WordValue& weight,NodeId* nid,int levelsup) const
{
	// propagate the feature down the tree
	
	// level at which the node must be stored in nid, if given
  	const int nid_level = m_L - levelsup;
  	if(nid_level <= 0 && nid != NULL) *nid = 0;	// root

  	NodeId final_id = 0;	// root
  	int current_level = 0;
	
	do{
		++current_level;
    	auto const  &nodes = m_nodes[final_id].children;
    	double best_d = std::numeric_limits<double>::max();
		for(const auto &id : nodes){
			double d = DescriptorsManipulator::distance(feature,m_nodes[id].descriptor);
			if(d < best_d){
				best_d = d;
				final_id = id;
      		}
    	}
		
		if(nid != NULL && current_level == nid_level) *nid = final_id;
	}while( !m_nodes[final_id].isLeaf() );
	
	// turn node id into word id
	word_id = m_nodes[final_id].word_id;
  	weight = m_nodes[final_id].weight;
}

void Vocabulary::transform(const cv::Mat& feature,WordId& word_id,WordValue& weight) const
{
	// propagate the feature down the tree
	
	NodeId final_id = 0;	// root
	if (feature.type() == CV_8U){
		do{
			auto const& nodes = m_nodes[final_id].children;
			uint64_t best_d = std::numeric_limits<uint64_t>::max();
          	int idx=0,bestidx=0;
           	for(const auto &id : nodes){
				// compute distance	
				uint64_t dist = DescriptorsManipulator::distance_8uc1(feature,m_nodes[id].descriptor);
              	if(dist < best_d){
					best_d = dist;
                  	final_id = id;
                  	bestidx=idx;
              	}
              	idx++;
          	}
		}while(!m_nodes[final_id].isLeaf());
   	}
	else{
		do{
			auto const& nodes = m_nodes[final_id].children;
		  	uint64_t best_d = std::numeric_limits<uint64_t>::max();
		  	int idx = 0, bestidx = 0;
		  	for(const auto &id : nodes){
				// compute distance
				uint64_t dist = DescriptorsManipulator::distance(feature, m_nodes[id].descriptor);
				if(dist < best_d){
					best_d = dist;
				  	final_id = id;
				  	bestidx = idx;
			  	}
			  	idx++;
		  	}
	  	}while(!m_nodes[final_id].isLeaf());
  	}
	
	// turn node id into word id
  	word_id = m_nodes[final_id].word_id;
  	weight = m_nodes[final_id].weight;
}

void Vocabulary::transform(const cv::Mat& feature,WordId &id) const
{
	WordValue weight;
	transform(feature,id,weight);
}

void Vocabulary::HKmeansStep(NodeId parent_id,const std::vector<cv::Mat>& descriptors,int current_level)
{
	if(descriptors.empty()) return;

    // features associated to each cluster
    std::vector<cv::Mat> clusters;
    std::vector<std::vector<unsigned int>> groups;
    clusters.reserve(m_k);
    groups.reserve(m_k);
	
	if((int)descriptors.size() <= m_k){
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
                initiateClusters(descriptors,clusters);
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
					DescriptorsManipulator::meanValue(cluster_descriptors,clusters[c]);
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
        NodeId id = m_nodes.size();
        m_nodes.emplace_back(Node(id));
        m_nodes.back().descriptor = clusters[i];
        m_nodes.back().parent = parent_id;
        m_nodes[parent_id].children.emplace_back(id);
    }

    // go on with the next level
    if(current_level < m_L){
        // iterate again with the resulting clusters
        const std::vector<NodeId>& children_ids = m_nodes[parent_id].children;
        for(unsigned int i = 0; i < clusters.size(); i++){
            NodeId id = children_ids[i];

            std::vector<cv::Mat> child_features;
            child_features.reserve(groups[i].size());

            std::vector<unsigned int>::const_iterator vit;
            for(vit = groups[i].begin(); vit != groups[i].end(); vit++){
                child_features.emplace_back(descriptors[*vit]);
            }

            if(child_features.size() > 1) HKmeansStep(id,child_features,current_level + 1);   
        }
    }
}

void Vocabulary::initiateClusters(const std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters) const
{
	initiateClustersKMpp(descriptors, clusters);
}

void Vocabulary::initiateClustersKMpp(const std::vector<cv::Mat>& pfeatures,std::vector<cv::Mat>& clusters) const
{
	clusters.resize(0);
  	clusters.reserve(m_k);
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

  	while((int)clusters.size() < m_k){
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

void Vocabulary::createWords()
{
	m_words.resize(0);
	if(!m_nodes.empty()){
		m_words.reserve((int)std::pow((double)m_k,(double)m_L) );
		
		auto nit = m_nodes.begin();	// ignore root
    	for(++nit; nit != m_nodes.end(); nit++){
			if(nit->isLeaf()){
				nit->word_id = m_words.size();
				m_words.emplace_back( &(*nit) );
      		}
    	}
  	}
}

void Vocabulary::setNodeWeights(const std::vector<std::vector<cv::Mat>>& training_features)
{
	const unsigned int NWords = m_words.size();
	const unsigned int NDocs = training_features.size();
	if(m_weighting == TF || m_weighting == BINARY){
		// idf part must be 1 always
    	for(unsigned int i = 0; i < NWords; i++) m_words[i]->weight = 1;
  	}
  	else if(m_weighting == IDF || m_weighting == TF_IDF){
		// IDF and TF-IDF: we calculte the idf path now
		std::vector<unsigned int> Ni(NWords, 0);
		std::vector<bool> counted(NWords, false);
		for(auto mit = training_features.begin(); mit != training_features.end(); mit++){
			std::fill(counted.begin(),counted.end(),false);
			for(auto fit = mit->begin(); fit < mit->end(); fit++){
				WordId word_id;
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
				m_words[i]->weight = std::log((double)NDocs/(double)Ni[i]);
      		}
    	}
  	}
}

void Vocabulary::load_fromtxt(const std::string& file_name)
{
	std::ifstream ifile(file_name);
    if(!ifile) throw std::runtime_error("Vocabulary:: load_fromtxt  Could not open file for reading: " + file_name);
    
	int n1, n2;
    {
		std::string str;
		getline(ifile,str);
		std::stringstream ss(str);
    	ss >> m_k >> m_L >> n1 >> n2;
    }

    if(m_k < 0 || m_k > 20 || m_L < 1 || m_L > 10 || n1 < 0 || n1 > 5 || n2 < 0 || n2 > 3){
		throw std::runtime_error( "Vocabulary loading failure: This is not a correct text file!" );
	}
    m_scoring = (ScoringType)n1;
    m_weighting = (WeightingType)n2;
    createScoringObject();
    
	// nodes
	int expected_nodes = (int)((std::pow((double)m_k,(double)m_L + 1) - 1)/(m_k - 1));
    m_nodes.reserve(expected_nodes);
    m_words.reserve(pow((double)m_k,(double)m_L + 1));
	m_nodes.resize(1);
    m_nodes[0].id = 0;

    int counter=0;
    while(!ifile.eof()){
		std::string snode;
        getline(ifile,snode);
        if(counter++ % 100 == 0) std::cerr <<".";
		if(snode.size() == 0) break;
        
		std::stringstream ssnode(snode);
		int nid = m_nodes.size();
		m_nodes.resize(m_nodes.size() + 1);
        m_nodes[nid].id = nid;

        int pid ;
        ssnode >> pid;
        m_nodes[nid].parent = pid;
        m_nodes[pid].children.emplace_back(nid);

        int nIsLeaf;
        ssnode >> nIsLeaf;

        // read until the end and add to data
        std::vector<float> data;
		data.reserve(100);
        
		float d;
        while(ssnode >> d) data.emplace_back(d);
        
		// the weight is the last
        m_nodes[nid].weight=data.back();
        data.pop_back();	// remove
        
		// the rest, to the descriptor
        m_nodes[nid].descriptor.create(1,data.size(),CV_8UC1);
        auto ptr = m_nodes[nid].descriptor.ptr<uchar>(0);
        for(auto d : data) *ptr++ = d;

        if(nIsLeaf > 0){
			int wid = m_words.size();
            m_words.resize(wid + 1);
			
			m_nodes[nid].word_id = wid;
            m_words[wid] = &m_nodes[nid];
        }
        else{
			m_nodes[nid].children.reserve(m_k);
        }
    }
}

/*
std::ostream& operator<<(std::ostream& os,const Vocabulary& voc)
{
	os << "Vocabulary: k = " << voc.getBranchingFactor()
       << ", L = " << voc.getDepthLevels()
       << ", Weighting = ";
	
	switch(voc.getWeightingType())
  	{
		case TF_IDF: 
			os << "tf-idf"; 
			break;

    	case TF: 
			os << "tf"; 
			break;

    	case IDF: 
			os << "idf"; 
			break;
    	
		case BINARY: 
			os << "binary"; 
			break;
  	}

  	os << ", Scoring = ";
  	switch(voc.getScoringType())
	{
    	case L1_NORM: 
			os << "L1-norm"; 
			break;

    	case L2_NORM: 
			os << "L2-norm"; 
			break;
    	
		case CHI_SQUARE: 
			os << "Chi square distance"; 
			break;
    	
		case KL: 
			os << "KL-divergence"; 
			break;
    	
		case BHATTACHARYYA: 
			os << "Bhattacharyya coefficient";
			break;
    	
		case DOT_PRODUCT: 
			os << "Dot product"; 
			break;
  	}
	
	os << ", Number of words = " << voc.size();
	return os;
}
*/

void Vocabulary::get_info()
{
	std::cout << "k = " << getBranchingFactor() << std::endl;
	std::cout << "L = " << getDepthLevels() << std::endl;
	
	std::cout << "Weighting = ";
	switch(getWeightingType())
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
	switch(getDescritorType())
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
	
	std::cout << "Number of words = " << size() << std::endl;
	std::cout << std::endl;

}