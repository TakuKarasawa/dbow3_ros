#ifndef SCORING_OBJECT_H_
#define SCORING_OBJECT_H_

#include <cfloat>

#include "dbow3/bow_vector/bow_vector.h"

namespace dbow3
{
// Base class of scoring functions
class GeneralScoring
{
public:
    virtual ~GeneralScoring() {}
    virtual double score(const BowVector& v,const BowVector& w) const = 0;
    virtual bool mustNormalize(LNorm &norm) const = 0;

	static const double LOG_EPS;
};

#define __SCORING_CLASS(NAME, MUSTNORMALIZE, NORM) NAME: public GeneralScoring \
{   public: \
    \
    virtual double score(const BowVector& v,const BowVector& w) const; \
    \
    \
    virtual inline bool mustNormalize(LNorm &norm) const  \
        { norm = NORM; return MUSTNORMALIZE; } \
}

/// L1 Scoring object
class __SCORING_CLASS(L1Scoring, true, L1);

/// L2 Scoring object
class __SCORING_CLASS(L2Scoring, true, L2);

/// Chi square Scoring object
class __SCORING_CLASS(ChiSquareScoring, true, L1);

/// KL divergence Scoring object
class __SCORING_CLASS(KLScoring, true, L1);

/// Bhattacharyya Scoring object
class __SCORING_CLASS(BhattacharyyaScoring, true, L1);

/// Dot product Scoring object
class __SCORING_CLASS(DotProductScoring, false, L1);

#undef __SCORING_CLASS

}   // namespace dbow3

#endif  // SCORING_OBJECT_H_
