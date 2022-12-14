#include "dbow3/query_results/query_results.h"

using namespace dbow3;

inline void QueryResults::scale_scores(double factor)
{
    for(QueryResults::iterator qit = begin(); qit != end(); qit++) qit->score *= factor;
}

void QueryResults::save_M(const std::string &filename) const
{
    std::fstream f(filename.c_str(),std::ios::out);
    QueryResults::const_iterator qit;
    for(qit = begin(); qit != end(); qit++){
        f << qit->id << " " << qit->score << std::endl;
    }
    f.close();
}
