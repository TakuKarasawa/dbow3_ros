#include "dbow3_ros/dbow3_demo.h"

using namespace dbow3;

DBoW3Demo::DBoW3Demo() :
	private_nh_("~")
{
	std::string mode;
    private_nh_.param("MODE",mode,{std::string("orb")});
    set_detector_mode(mode);

    private_nh_.param("DIR_PATH",DIR_PATH_,{std::string("")});
}

void DBoW3Demo::set_detector_mode(std::string mode)
{
	if(mode == "orb") detector_ = cv::ORB::create();
    else if(mode == "brisk") detector_ = cv::BRISK::create();
    else if(mode == "akaze") detector_ = cv::AKAZE::create();
    else{
		std::cerr << "No applicable mode. Please select 'orb', 'brisk' or 'akaze'" << std::endl;
        std::cerr  << "Set 'orb" << std::endl;
        detector_ = cv::ORB::create();
    }
}

void DBoW3Demo::create_vocabulary(const std::vector<cv::Mat>& features)
{
	// branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
	Vocabulary voc(k,L,weight,score);

    std::cout << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
    voc.create(features);
    std::cout << "... done!" << std::endl;

    std::cout << "Vocabulary information: " << std::endl;
	voc.get_info();

    std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;
    BowVector v1, v2;
    for(size_t i = 0; i < features.size(); i++){
		voc.transform(features[i], v1);
        for(size_t j = 0; j < features.size(); j++){
            voc.transform(features[j],v2);
            double score = voc.score(v1,v2);
            std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
        }
    }

    // save the vocabulary to disk
    std::cout << std::endl << "Saving vocabulary..." << std::endl;
    voc.save("small_voc.yml.gz");
    std::cout << "Done" << std::endl;
}

void DBoW3Demo::create_database(std::vector<cv::Mat>& features)
{
	std::cout << "Creating a small database..." << std::endl;

    // load the vocabulary from disk
    Vocabulary voc("small_voc.yml.gz");

    Database db(voc,false,0);	// false = do not use direct index

    // add images to the database
    for(size_t i = 0; i < features.size(); i++) db.add(features[i]);
    std::cout << "... done!" << std::endl;

    std::cout << "Database information: " << std::endl;
	db.get_info();

	/*
    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
	*/
}

std::vector<cv::Mat> DBoW3Demo::load_features()
{
	std::cout << "Extracting features ..." << std::endl;
	std::vector<cv::Mat> features;
	for(size_t i = 0; ; i++){
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		std::string file_path = DIR_PATH_ + "image" + std::to_string(i) + ".jpg";
        cv::Mat image = cv::imread(file_path,0);
        if(image.empty()) break;
		std::cout << "file name: " << file_path << std::endl;
        detector_->detectAndCompute(image,cv::Mat(),keypoints,descriptors);
        features.emplace_back(descriptors);
    }
    std::cout << std::endl;
    return features;
}

void DBoW3Demo::process()
{
	std::vector<cv::Mat> features = load_features();
	create_vocabulary(features);
	create_database(features);
}

int main(int argc,char** argv)
{
	ros::init(argc,argv,"dbow3_demo");
	DBoW3Demo dbow3_demo;
	dbow3_demo.process();
	return 0;
}