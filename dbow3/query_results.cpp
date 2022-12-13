#include "dbow3/query_results/query_results.h"

using namespace dbow3;

inline void QueryResults::scaleScores(double factor)
{
	for(QueryResults::iterator qit = begin(); qit != end(); qit++) qit->Score *= factor;
}

void QueryResults::saveM(const std::string &filename) const
{
	std::fstream f(filename.c_str(),std::ios::out);
	QueryResults::const_iterator qit;
	for(qit = begin(); qit != end(); qit++){
		f << qit->Id << " " << qit->Score << std::endl;
  	}
  	f.close();
}

