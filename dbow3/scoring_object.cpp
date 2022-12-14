#include "dbow3/scoring_object/scoring_object.h"

using namespace dbow3;

const double GeneralScoring::LOG_EPS = std::log(DBL_EPSILON);

double L1Scoring::score(const BowVector& v1,const BowVector& v2) const
{
    BowVector::const_iterator v1_it, v2_it;
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    v1_it = v1.begin();
    v2_it = v2.begin();

    double score = 0;
    while(v1_it != v1_end && v2_it != v2_end){
        const double& vi = v1_it->second;
        const double& wi = v2_it->second;

        if(v1_it->first == v2_it->first){
            score += std::fabs(vi - wi) - std::fabs(vi) - std::fabs(wi);

            // move v1 and v2 forward
            v1_it++;
            v2_it++;
        }
        else if(v1_it->first < v2_it->first){
            // move v1 forward
            v1_it = v1.lower_bound(v2_it->first);
        }
        else{
            // move v2 forward
            v2_it = v2.lower_bound(v1_it->first);
        }
    }
    score = -score/2.0;
    return score;
}

double L2Scoring::score(const BowVector& v1,const BowVector& v2) const
{
    BowVector::const_iterator v1_it, v2_it;
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    v1_it = v1.begin();
    v2_it = v2.begin();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end){
        const double& vi = v1_it->second;
        const double& wi = v2_it->second;

        if(v1_it->first == v2_it->first){
            score += vi * wi;

            // move v1 and v2 forward
            v1_it++;
            v2_it++;
        }
        else if(v1_it->first < v2_it->first){
            // move v1 forward
            v1_it = v1.lower_bound(v2_it->first);
        }
        else{
            // move v2 forward
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    if(score >= 1){
        // rounding errors
        score = 1.0;
    }
    else score = 1.0 - std::sqrt(1.0 - score);  // [0..1]

    return score;
}

double ChiSquareScoring::score(const BowVector& v1,const BowVector& v2) const
{
    BowVector::const_iterator v1_it, v2_it;
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    v1_it = v1.begin();
    v2_it = v2.begin();

    double score = 0;

    // all the items are taken into account
    while(v1_it != v1_end && v2_it != v2_end){
        const double& vi = v1_it->second;
        const double& wi = v2_it->second;

        if(v1_it->first == v2_it->first){
            if(vi + wi != 0.0) score += vi*wi/(vi + wi);

            // move v1 and v2 forward
            v1_it++;
            v2_it++;
        }
        else if(v1_it->first < v2_it->first){
            // move v1 forward
            v1_it = v1.lower_bound(v2_it->first);
        }
        else{
            // move v2 forward
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    // this takes the -4 into account
    score = 2.*score;   // [0..1]

    return score;
}

double KLScoring::score(const BowVector& v1,const BowVector& v2) const
{
    BowVector::const_iterator v1_it, v2_it;
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    v1_it = v1.begin();
    v2_it = v2.begin();

    double score = 0;

    // all the items or v are taken into account
    while(v1_it != v1_end && v2_it != v2_end){
        const double& vi = v1_it->second;
        const double& wi = v2_it->second;

        if(v1_it->first == v2_it->first){
            if(vi != 0 && wi != 0) score += vi*std::log(vi/wi);

            // move v1 and v2 forward
            v1_it++;
            v2_it++;
        }
        else if(v1_it->first < v2_it->first){
            // move v1 forward
            score += vi*(std::log(vi) - LOG_EPS);
            v1_it++;
        }
        else{
            // move v2_it forward, do not add any score
            v2_it = v2.lower_bound(v1_it->first);
        }
    }

    // sum rest of items of v
    for( ; v1_it != v1_end; v1_it++){
        if(v1_it->second != 0) score += v1_it->second*(std::log(v1_it->second) - LOG_EPS);
    }
    return score;
}

double BhattacharyyaScoring::score(const BowVector& v1,const BowVector& v2) const
{
    BowVector::const_iterator v1_it, v2_it;
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    v1_it = v1.begin();
    v2_it = v2.begin();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end){
        const double& vi = v1_it->second;
        const double& wi = v2_it->second;

        if(v1_it->first == v2_it->first){
            score += std::sqrt(vi * wi);

            // move v1 and v2 forward
            v1_it++;
            v2_it++;
        }
        else if(v1_it->first < v2_it->first){
            // move v1 forward
            v1_it = v1.lower_bound(v2_it->first);
        }
        else{
            // move v2 forward
            v2_it = v2.lower_bound(v1_it->first);
        }
  }
  return score;
}

double DotProductScoring::score(const BowVector& v1,const BowVector& v2) const
{
    BowVector::const_iterator v1_it, v2_it;
    const BowVector::const_iterator v1_end = v1.end();
    const BowVector::const_iterator v2_end = v2.end();

    v1_it = v1.begin();
    v2_it = v2.begin();

    double score = 0;
    while(v1_it != v1_end && v2_it != v2_end){
        const double& vi = v1_it->second;
        const double& wi = v2_it->second;

        if(v1_it->first == v2_it->first){
            score += vi*wi;

            // move v1 and v2 forward
            v1_it++;
            v2_it++;
        }
        else if(v1_it->first < v2_it->first){
            // move v1 forward
            v1_it = v1.lower_bound(v2_it->first);
        }
        else{
            // move v2 forward
            v2_it = v2.lower_bound(v1_it->first);
        }
    }
    return score;
}
