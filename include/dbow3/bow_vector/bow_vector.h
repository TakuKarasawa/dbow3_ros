#ifndef BOW_VECTOR_H_
#define BOW_VECTOR_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>

#include "dbow3/bow_vector/l_norm.h"
#include "dbow3/bow_vector/weighting_type.h"
#include "dbow3/bow_vector/scoring_type.h"

namespace dbow3
{
class BowVector : public std::map<unsigned int,double>
{
public:
    BowVector();
    ~BowVector();

    void add_weight(unsigned int id,double v);
    void add_if_not_exist(unsigned id,double v);
    void normalize(LNorm norm_type);
    void save_M(const std::string& filename,size_t W) const;
    uint64_t get_signature() const;

    // stream utils
    void to_stream(std::ostream& str) const;
    void from_stream(std::istream& str);

    // operator
    friend std::ostream& operator<<(std::ostream& out,const BowVector& v);
};
}   // namespace dbow3

#endif  // BOW_VECTOR_H_
