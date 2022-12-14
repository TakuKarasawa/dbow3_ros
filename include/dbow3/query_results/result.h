#ifndef RESULT_H_
#define RESULT_H_

#include <fstream>
#include <ostream>

namespace dbow3
{
// Single result of a query
class Result
{
public:
    inline Result() {}
    inline Result(unsigned int _id,double _score) :
        id(_id), score(_score) {}

    inline bool operator<(const Result &r) const { return this->score < r.score; }
    inline bool operator>(const Result &r) const { return this->score > r.score; }
    inline bool operator==(unsigned int id) const { return this->id == id; }
    inline bool operator<(double s) const { return this->score < s; }
    inline bool operator>(double s) const { return this->score > s; }

    static inline bool gt(const Result& a,const Result& b) { return a.score > b.score; }
    inline static bool ge(const Result& a,const Result& b) { return a.score > b.score; }
    static inline bool geq(const Result& a,const Result& b) { return a.score >= b.score; }
    static inline bool geqv(const Result& a,double s) { return a.score >= s; }
    static inline bool ltId(const Result& a,const Result& b) { return a.id < b.id; }

    friend std::ostream & operator<<(std::ostream& os, const Result& ret )
    {
        os << "<EntryId: " << ret.id << ", Score: " << ret.score << ">";
        return os;
    }

    unsigned int id;    // Entry id
    int words;          // words in common
    double score;       // Score obtained
    double bhat_score;
    double chi_score;

    // only done by ChiSq and BCThresholding
    double sum_common_vi;
    double sum_common_wi;
    double expected_chi_score;
};
}   // namespace dbow3

#endif  // RESULT_H_
