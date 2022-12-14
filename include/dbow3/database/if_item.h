#ifndef IF_ITEM_H_
#define IF_ITEM_H_

namespace dbow3
{
class IFItem
{
public:
    IFItem() {}
    IFItem(unsigned int eid,double wv) :
        entry_id(eid), word_weight(wv) {}

    inline bool operator==(unsigned int eid) const
    {
        return entry_id == eid;
    }

    unsigned int entry_id;  // Entry id
    double word_weight;     // Word weight in this entry

private:
};
}   // namespace dbow3

#endif  // IF_ITEM_H_
