#ifndef SCORING_TYPE_H_
#define SCORING_TYPE_H_

namespace dbow3
{
// Scoring type
enum ScoringType
{
    L1_NORM,
    L2_NORM,
    CHI_SQUARE,
    KL,
    BHATTACHARYYA,
    DOT_PRODUCT
};
}   // namespace dbow3

#endif  // SCORING_TYPE_H_
