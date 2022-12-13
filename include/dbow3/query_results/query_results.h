#ifndef QUERY_RESULTS_H_
#define QUERY_RESULTS_H_

#include <fstream>
#include <ostream>
#include <vector>

namespace dbow3
{
/// Id of entries of the database
typedef unsigned int EntryId;

// Single result of a query
class Result
{
public:
	/// Entry id
  	EntryId Id;
  
  	/// Score obtained
  	double Score;
  
  	int nWords; // words in common
  
  	double bhatScore, chiScore;
  
  	// only done by ChiSq and BCThresholding 
  	double sumCommonVi;
  	double sumCommonWi;
  	double expectedChiScore;

  	inline Result() {}
  	inline Result(EntryId _id,double _score) : 
		Id(_id), Score(_score) {}
	
	inline bool operator<(const Result &r) const { return this->Score < r.Score; }
	inline bool operator>(const Result &r) const { return this->Score > r.Score; }
	inline bool operator==(EntryId id) const { return this->Id == id; }
	inline bool operator<(double s) const { return this->Score < s; }
	inline bool operator>(double s) const { return this->Score > s; }
	
	static inline bool gt(const Result& a,const Result& b) { return a.Score > b.Score; }
	inline static bool ge(const Result& a,const Result& b) { return a.Score > b.Score; }
	static inline bool geq(const Result& a,const Result& b) { return a.Score >= b.Score; }
	static inline bool geqv(const Result& a,double s) { return a.Score >= s; }
	static inline bool ltId(const Result& a,const Result& b) { return a.Id < b.Id; }
	
	friend std::ostream & operator<<(std::ostream& os, const Result& ret )
	{
		os << "<EntryId: " << ret.Id << ", Score: " << ret.Score << ">";
  		return os;
	}
};

// Multiple results from a query
class QueryResults: public std::vector<Result>
{
public:
	inline void scaleScores(double factor);
	void saveM(const std::string &filename) const;
	
	friend std::ostream & operator<<(std::ostream& os,const QueryResults& ret)
	{
		if(ret.size() == 1) os << "1 result:" << std::endl;
		else os << ret.size() << " results:" << std::endl;
		
		QueryResults::const_iterator rit;
		for(rit = ret.begin(); rit != ret.end(); rit++){
			os << *rit;
			if(rit + 1 != ret.end()) os << std::endl;
  		}
  		return os;
	}
};

} // namespace TemplatedBoW
  
#endif	// QUERY_RESULTS_H_