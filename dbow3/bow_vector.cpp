#include "dbow3/bow_vector/bow_vector.h"

using namespace dbow3;

BowVector::BowVector() {}

BowVector::~BowVector() { }

void BowVector::add_weight(unsigned int id,double v)
{
    BowVector::iterator vit = this->lower_bound(id);
    if(vit != this->end() && !(this->key_comp()(id, vit->first))) vit->second += v;
    else this->insert(vit,BowVector::value_type(id,v));
}

void BowVector::add_if_not_exist(unsigned id,double v)
{
    BowVector::iterator vit = this->lower_bound(id);
    if(vit == this->end() || (this->key_comp()(id, vit->first))) this->insert(vit,BowVector::value_type(id,v));
}

void BowVector::normalize(LNorm norm_type)
{
    double norm = 0.0;
    BowVector::iterator it;
    if(norm_type == dbow3::L1){
        for(it = begin(); it != end(); it++) norm += std::fabs(it->second);
    }
    else{
        for(it = begin(); it != end(); it++) norm += it->second * it->second;
        norm = std::sqrt(norm);
    }

    if(norm > 0.0){
        for(it = begin(); it != end(); it++) it->second /= norm;
    }
}

void BowVector::save_M(const std::string& filename, size_t W) const
{
    std::fstream f(filename.c_str(), std::ios::out);

    unsigned int last = 0;
    BowVector::const_iterator bit;
    for(bit = this->begin(); bit != this->end(); bit++){
        for( ; last < bit->first; last++){
            f << "0 ";
        }
        f << bit->second << " ";
        last = bit->first + 1;
    }

    for( ; last < (unsigned int)W; last++) f << "0 ";
    f.close();
}

uint64_t BowVector::get_signature() const
{
    uint64_t sig=0;
    for(auto ww : *this) sig += ww.first + 1e6*ww.second;
    return sig;
}

void BowVector::to_stream(std::ostream& str) const
{
    uint32_t s=size();
    str.write((char*)&s,sizeof(s));
    for(auto d : *this){
        str.write((char*)&d.first,sizeof(d.first));
        str.write((char*)&d.second,sizeof(d.second));
    }
}

void BowVector::from_stream(std::istream& str)
{
    clear();
    uint32_t s;

    str.read((char*)&s,sizeof(s));
    for(int i = 0; i < s; i++){
        unsigned int wid;
        double wv;
        str.read((char*)&wid,sizeof(wid));
        str.read((char*)&wv,sizeof(wv));
        insert(std::make_pair(wid,wv));
    }
}

std::ostream& operator<< (std::ostream& out,const BowVector& v)
{
    BowVector::const_iterator vit;
    std::vector<unsigned int>::const_iterator iit;
    unsigned int i = 0;
    const size_t N = v.size();
    for(vit = v.begin(); vit != v.end(); vit++, i++){
        out << "<" << vit->first << ", " << vit->second << ">";
        if(i < N-1) out << ", ";
    }
    return out;
}
