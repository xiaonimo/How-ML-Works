#ifndef MULTI_LR_HPP
#define MULTI_LR_HPP
#include "lr.hpp"
#include "datatype.hpp"

class MLR {
public:
    MLR(points_t& _train_x, std::vector<index_t>& _train_y, const unsigned _n_class):
    train_x(_train_x), train_y(_train_y), n_class(_n_class){
        //classifiers = std::vector<std::vector<LR>>(n_class, std::vector<LR>(n_class, LR(points_t(), std::vector<index_t>())));
        prob.assign(n_class, 0);
    }
    void fit();
    std::vector<std::vector<index_t>> predict_prob(const points_t&);

public:
    std::vector<LR> classifiers;

private:
    void get_AB_data(points_t&, std::vector<index_t>&, index_t, index_t);

private:
    points_t& train_x;
    std::vector<index_t>& train_y;
    const unsigned n_class;
    points_t AB_train_x;
    std::vector<index_t> AB_train_y;
    std::vector<unsigned> prob;
};

std::vector<std::vector<index_t>>
MLR::predict_prob(const points_t & test_x) {
    std::vector<std::vector<index_t>> res;
    for (index_t i=0; i<test_x.size(); ++i) {
        prob.assign(n_class, 0);
        for (index_t A=0; A<n_class-1; ++A) {
            for (index_t B=A; B<n_class-1; ++B) {
                double pb = classifiers[A*(n_class-1) + B].predict_prob(test_x[i]);
                if (pb>0.5) {
                    prob[A]++;
                } else {
                    prob[B+1]++;
                }
            }
        }
        res.push_back(prob);
    }
    return res;
}

//A:1   B:0
void
MLR::get_AB_data(points_t &x, std::vector<index_t> &y, index_t A, index_t B) {
    x.clear();
    y.clear();
    for (index_t i=0; i<train_x.size(); ++i) {
        if (index_t(train_y[i])==A) {
            x.push_back(train_x[i]);
            y.push_back(1);
        } else if (index_t(train_y[i])==B) {
            x.push_back(train_x[i]);
            y.push_back(0);
        } else {}
    }
}

void
MLR::fit() {
    for (index_t classA = 0; classA<n_class-1; ++classA) {
        for (index_t classB=0; classB<n_class-1; ++classB) {
            if (classB >= classA) {
                std::cout << "classification:" << classA << ":" << classB <<std::endl;
                get_AB_data(AB_train_x, AB_train_y, classA, classB+1);
                LR blr(AB_train_x, AB_train_y);
                blr.fit();
                classifiers.push_back(blr);
            } else {
                classifiers.push_back(classifiers[0]);
            }
        }
    }
}


#endif // MULTI_LR_HPP
