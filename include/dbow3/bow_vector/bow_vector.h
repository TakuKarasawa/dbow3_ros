#ifndef BOW_VECTOR_H_
#define BOW_VECTOR_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>

#if _WIN32
#include <cstdint>
#endif

namespace dbow3
{
/// Id of words
typedef unsigned int WordId;

/// Value of a word
typedef double WordValue;

/// Id of nodes in the vocabulary tree
typedef unsigned int NodeId;

/// L-norms for normalization
enum LNorm
{
  L1,
  L2
};

/// Weighting type
enum WeightingType
{
  TF_IDF,
  TF,
  IDF,
  BINARY
};

/// Scoring type
enum ScoringType
{
  L1_NORM,
  L2_NORM,
  CHI_SQUARE,
  KL,
  BHATTACHARYYA,
  DOT_PRODUCT
};

/// Vector of words to represent images
class BowVector : public std::map<WordId,WordValue>
{
public:
	BowVector();
	~BowVector();

	void addWeight(WordId id, WordValue v);
	void addIfNotExist(WordId id, WordValue v);
	void normalize(LNorm norm_type);
	friend std::ostream& operator<<(std::ostream &out, const BowVector &v);
	void saveM(const std::string &filename, size_t W) const;
    uint64_t getSignature()const;
    void toStream(std::ostream &str)const;
    void fromStream(std::istream &str);

private:
};
}	// namespace dbow3

#endif	// BOW_VECTOR_H_