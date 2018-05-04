#ifndef LR_HPP
#define LR_HPP

#include "datatype.hpp"
#include "mnist.hpp"
#include <random>
#include <ctime>


//二分类
class LR{
public:
    //LR(points_t&x, std::vector<index_t>&y):train_x(x), train_y(y), n_input(0), n_epoch(0), n_batch(0), n_step(0), learning_rate(0), min_loss(0){}
    LR(points_t& _train_x, std::vector<index_t>& _train_y, const unsigned _epoch=20, const unsigned _batch=5, param_t _learning_rate=0.001, param_t _min_loss=0.01,
       bool _verbose=true):
        train_x(_train_x), train_y(_train_y), n_input(train_x[0].size()), n_epoch(_epoch), n_batch(_batch),
        n_step(train_x.size()/n_batch), learning_rate(_learning_rate), min_loss(_min_loss), verbose(_verbose) {

        weights.assign(n_input, 0);
        batch_dw.assign(n_input, 0);
        bias = 0;
        init_weights();
    }
    void fit();
    double predict_prob(const point_t&);
    vec_t predict_prob(const points_t&);
    LR& operator =(const LR&);

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
    vec_t weights;
    vec_t batch_dw;
    param_t bias;
    param_t db;

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

LR&
LR::operator =(const LR& p) {
    /*
    this->batch_dw = p.batch_dw;
    this->bias = p.bias;
    this->db = p.db;
    this->learning_rate = p.learning_rate;
    this->min_loss = p.min_loss;
    this->n_batch = p.n_batch;
    this->n_epoch = p.n_epoch;
    this->n_input = p.n_input;
    this->n_step = p.n_step;
    this->train_x = p.train_x;
    this->train_y = p.train_y;
    this->verbose = p.verbose;*/
    this->weights = p.weights;
    return *this;
}

void
LR::backword_flow() {
    db += hx - y;
    //std::cout << y <<"\t" << hx << "\t" << log(hx) << "\t" << log(1-hx) << std::endl;
    //getchar();
    //cur_loss += y*log(hx)+(1-y)*log(1-hx);
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
    /*
    else if (int(train_y[cur_index][1]) == 1) y=1;
    else if (int(train_y[cur_index][2]) == 1) y=1;
    else if (int(train_y[cur_index][3]) == 1) y=1;
    else if (int(train_y[cur_index][4]) == 1) y=1;
    */
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
