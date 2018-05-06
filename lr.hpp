#ifndef LR_HPP
#define LR_HPP

#include "datatype.hpp"
#include "func.hpp"
#include <random>
#include <ctime>


//二分类
class LR {
public:
    LR(points_t& _train_x, std::vector<index_t>& _train_y, const unsigned _n_epoch=10, const unsigned _n_batch=3,
       param_t _learning_rate=0.001, param_t _min_loss=0.01, bool _verbose=true):
        train_x(_train_x), train_y(_train_y), n_input(train_x[0].size()),
        n_epoch(_n_epoch), n_batch(_n_batch), n_step(train_x.size()/n_batch),
        learning_rate(_learning_rate), min_loss(_min_loss), verbose(_verbose) {

        weights.assign(n_input, 0);
        batch_dw.assign(n_input, 0);
        bias = 0; db=0;
        init_weights();
    }
    void fit();
    double predict_prob(const point_t&);
    vec_t  predict_prob(const points_t&);

public:
    points_t &train_x;
    std::vector<index_t>& train_y;
    const unsigned n_input;
    const unsigned n_epoch;
    const unsigned n_batch;
    const unsigned n_step;
    const param_t learning_rate;
    const param_t min_loss;
    bool verbose;
    vec_t weights, batch_dw;
    param_t bias, db;

private:
    void init_weights();
    void forword_flow();
    void forword_flow(const point_t&);
    void backword_flow();
    void update_weights();
    void set_XY(index_t);

private:
    data_t hx;
    data_t y;
    index_t cur_index;
    param_t cur_loss=0, epoch_loss=0;
};

void
LR::backword_flow() {
    db += hx - y;
    cur_loss += std::pow(hx, y)*std::pow(1-hx, 1-y);
    for (index_t i=0; i<n_input; ++i) {
        batch_dw[i] += (hx-y)*train_x[cur_index][i];
    }
}

void
LR::update_weights() {
    bias -= learning_rate*db/n_batch;
    db = 0.;
    for (index_t i=0; i<n_input; ++i) {
        weights[i] -= learning_rate*batch_dw[i]/n_batch;
        batch_dw[i] = 0.;
    }
}

void
LR::set_XY(index_t index) {
    cur_index = index;
    if (int(train_y[cur_index]) == 1) y=1;
    else y=0;
}

void
LR::forword_flow(const point_t& p) {
    double z = 0.;
    for (index_t i=0; i<n_input; ++i) {
        z += weights[i]*p[i];
    }
    z += bias;
    hx = 1/(1+exp(-z));
}

void
LR::forword_flow() {
    double z = 0.;
    for (index_t i=0; i<n_input; ++i) {
        z += weights[i]*train_x[cur_index][i];
    }
    z += bias;
    hx = 1/(1+exp(-z));
}

void
LR::init_weights() {
    std::mt19937 gen;
    std::normal_distribution<double> nor(-0.01, 0.01);
    for (auto &e:weights) e=nor(gen);
    bias = nor(gen);
}

void
LR::fit() {
    for (index_t e=0; e<n_epoch; ++e) {
        epoch_loss = 0;
        auto t1 = clock();
        nn::random_shuffle(train_x, train_y);
        for (index_t s=0; s<n_step; ++s) {
            index_t itr = 0;
            cur_loss = min_loss+1;
            while (itr++ < 500 && cur_loss>min_loss) {
                cur_loss = 0;
                update_weights();
                for (index_t b=0; b<n_batch; ++b) {
                    set_XY(s*n_batch + b);
                    forword_flow();
                    backword_flow();
                }
            }
            epoch_loss += cur_loss;
        }
        auto t2 = clock();
        if (verbose) {
            std::cout << "epoch:" << e+1 << " loss:" <<epoch_loss/n_step << " time:" << t2-t1 << "ms" << std::endl;
        }
    }
}

double
LR::predict_prob(const point_t &p) {
    forword_flow(p);
    return hx;
}

vec_t
LR::predict_prob(const points_t &test_x) {
    vec_t res;
    for (index_t i=0; i<test_x.size(); ++i) {
        //forword_flow(test_x[i]);
        double pb = predict_prob(test_x[i]);
        res.push_back(pb);
    }
    return res;
}

#endif // LR_HPP
