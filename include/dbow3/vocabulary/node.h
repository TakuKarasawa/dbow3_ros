#ifndef NODE_H_
#define NODE_H_

#include <vector>

#include <opencv2/opencv.hpp>

namespace dbow3
{
// Tree node
class Node
{
public:
    Node() :
        id(0), weight(0), parent(0), word_id(0) {}

    Node(unsigned int _id) :
        id(_id), weight(0), parent(0), word_id(0) {}

    inline bool is_leaf() const { return children.empty(); }

    unsigned int id;                    // Node id
    double weight;                      // Weight if the node is a word
    std::vector<unsigned int> children; // Children
    unsigned int parent;                // Parent node (undefined in case of root)
    cv::Mat descriptor;                 // Node descriptor
    unsigned int word_id;               // Word id if the node is a word

private:
};
}   // namespace dbow3

#endif  // NODE_H_
