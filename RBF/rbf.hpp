#ifndef RBF_HPP
#define RBF_HPP

#include <iomanip>
#include <random>
#include <initializer_list>
#include "datatype.hpp"
#include "kmeans.hpp"

class rbf {
public:
    rbf(const points_t& _train_x, const points_t& _train_y, std::initializer_list _net, unsigned _epoch, unsigned _batch, bool _verbose=true):
       train_x(_train_x), train_y(_train_y), n_input(_net[0]), n_hidden(_net[1]), n_output(_net[2]),
       epoch(_epoch), batch(_batch), step(train_x.size()/batch), verbose(_verbose){

        weights = points_t(n_hidden, point_t(n_output));
        bias = vec_t(n_output);
        centers = points_t(n_hidden, point_t(n_input));
        gammas = vec_t(n_hidden);
        o1 = std::vector<double>(n_hidden);
        o2 = std::vector<double>(n_output);
        init_weights_bias();
    }
    void fit();
    indexes_t predict(const points_t&);
    std::vector<vec_t> predict_regression(const points_t&);

private:
    void set_X(index_t x) {x_index = x;}
    void set_Y(index_t y) {y_index = y;}
    void forword_flow();
    void _forword_flow1();
    void _forword_flow2();
    void backword_flow();
    void update_weights();
    data_t gaussion(const point_t&, const point_t&, double);
    data_t dist(const point_t&, const point_t&);
    index_t argmax(const point_t&);
    void init_weights();

private:
    index_t x_index, y_index;
    const points_t& train_x;
    const points_t& train_y;
    unsigned n_input, n_hidden, n_output;
    points_t centers;
    vec_t gammas;
    std::vector<vec_t> weights;
    vec_t bias;
    vec_t o1, o2; //隐含层和输出层的输出结果
    auto batch_o1 = points_t(batch, point_t(n_hidden, 0));
    auto batch_dw = points_t(n_hidden, point_t(n_output, 0));
    auto batch_db = vec_t(n_output, 0);
    auto batch_dc = points_t(n_hidden, point_t(n_input, 0));
    auto batch_dr = vec_t(n_hidden, 0);

public:
    double cur_loss;
    bool verbose;
    double learning_rate = 0.1;
    double min_loss = 0.01;
    unsigned epoch;
    unsigned batch;
    unsigned step;
};

void
rbf::init_weights_bias() {
    std::mt19937 gen;
    std::normal_distribution<double> normal(0, 0.0001);
    for(auto &_c:centers) for(auto &_e:_c) _e = normal(gen);
    for(auto &_e:gammas) _e = normal(gen);
    for(auto &_w:weights) for(auto &_e:_w) _e = normal(gen);
    for(auto &_e:bias) _e = normal(gen);
}

index_t
rbf::argmax(const point_t &p) {
    index_t res = 0;
    double _max_v = p[res];

    for (index_t i=1; i<p.size(); ++i) {
        if (p[i] < _max_v) continue;
        _max_v = p[i];
        res = i;
    }
    return res;
}

double
rbf::L2_dist(const point_t &p1, const point_t &p2) {
    double _dist = 0.;
    for (index_t i=0; i<p1.size(); ++i) _dist += pow(p1[i]-p2[i], 2);
    return _dist;
}

double
rbf::gaussion(const point_t &p1, const point_t &p2, double r) {
    return std::exp(-get_dist(p1, p2)/(2*(r*r)));
}

void
rbf::_forword_flow1() {
    for (index_t i=0; i<n_hidden; ++i) {
        o1[i] = gaussion(train_x[x_index], centers[i], gammas[i]);
    }
}

void
rbf::_forword_flow2() {
    for (index_t i=0; i<n_output; ++i) {
        double sum = 0.;
        for (index_t j=0; j<n_hidden; ++j) {
            sum += o1[j]*weights[j][i];
        }
        o2[i] = sum + bias[i];
    }
}

void
rbf::forword_flow() {
    _forword_flow1();
    _forword_flow2();
}

void
rbf::backword_flow() {
    // update bias
    for (index_t i=0; i<n_output; ++i) {
        bias[i] -= learning_rate*(o2[i]-train_y[y_index][i]);
    }

    //update weights
    for (index_t i=0; i<n_hidden; ++i) {
        for (index_t j=0; j<n_output; ++j) {
            weights[i][j] -= learning_rate*(o2[j]-train_y[y_index][j])*o1[i];
        }
    }
}

void
rbf::update_weights() {

}

void
rbf::fit() {
    double epoch_loss = 0;
    for (index_t e=0; e<epoch; ++e) {
        epoch_loss = 0;
        auto t1 = clock();

        for (index_t s=0; s<step; ++s) {
            //先保存中间结果
            for (index_t b=0; b<batch; ++b) {
                set_X(batch*s+b);
                _forword_flow1();
                batch_o1[b] = o1;
            }

            //开始迭代
            int __itr = 0;
            cur_loss = min_loss+1;
            while (__itr++ < 500 && cur_loss>min_loss) {
                //更新权重
                for (index_t i=0; i<n_output; ++i) {
                    bias[i] -= learning_rate*batch_db[i];
                }
                for (index_t i=0; i<n_hidden; ++i) {
                    for (index_t j=0; j<n_output; ++j) {
                        weights[i][j] -= learning_rate*batch_dw[i][j];
                    }
                }

                //清空batch_dw
                for (auto &_dw:batch_dw) for (auto &_e:_dw) _e = 0.;
                for (auto &_e:batch_db) _e=0.;
                cur_loss = 0;

                //计算敏感项
                for (index_t b=0; b<batch; ++b) {
                    o1 = batch_o1[b];
                    _forword_flow2();
                    for (index_t i=0; i<n_output; ++i) {
                        batch_db[i] += o2[i] - train_y[batch*s+b][i];
                        cur_loss += std::pow(o2[i] - train_y[batch*s+b][i], 2);
                        for (index_t j=0; j<n_hidden; ++j) {
                            batch_dw[j][i] += batch_db[i]*o1[j];
                        }
                    }
                }
                for (index_t i=0; i<n_output; ++i) {
                    batch_db[i] /= batch;
                    for (index_t j=0; j<n_hidden; ++j) {
                        batch_dw[j][i] /= batch;
                    }
                }

                //std::cout << cur_loss <<std::endl;
            }//end while
            //std::cout << "step:" << s << " loss:" <<cur_loss <<std::endl;
            //if (s==step-1 || s==0) std::cout << weights[0][0] << " " <<weights[1][0] <<std::endl;
            epoch_loss += cur_loss;
        }//end for(step)
        //std::cout << weights[0][0] << " ------------------------/" <<weights[1][0] <<std::endl;
        auto t2 = clock();
        std::cout << "epoch:" << e+1 << " loss:"<<epoch_loss/step <<" time:"<<(t2-t1)/double(CLOCKS_PER_SEC) << std::endl;
    }
}

std::vector<std::size_t>
rbf::predict(const points_t &test_x) {
    std::vector<std::size_t> res;
    /*
    for (index_t i=0; i<test_x.size(); ++i) {
        set_X(i);
        forword_flow();
        res.push_back(argmax(o2));
    }*/
    auto _y = predict_regression(test_x);
    for (auto _o:_y) res.push_back(argmax(_o));
    return res;
}

std::vector<point_t>
rbf::predict_regression(const points_t &test_x) {
    std::vector<point_t> res;
    for (index_t i=0; i<test_x.size(); ++i) {
        set_X(i);
        forword_flow();
        res.push_back(o2);
    }
    return res;
}

#endif // RBF_H








