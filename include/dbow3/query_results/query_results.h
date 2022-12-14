#ifndef QUERY_RESULTS_H_
#define QUERY_RESULTS_H_

#include <vector>

#include "dbow3/query_results/result.h"

namespace dbow3
{
// Multiple results from a query
class QueryResults: public std::vector<Result>
{
public:
    inline void scale_scores(double factor);
    void save_M(const std::string &filename) const;

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

}   // namespace TemplatedBoW

#endif  // QUERY_RESULTS_H_
