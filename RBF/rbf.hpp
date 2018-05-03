#ifndef RBF_HPP
#define RBF_HPP

#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <initializer_list>
#include "datatype.hpp"
#include "kmeans.hpp"
#include "mnist.hpp"

namespace std {
bool is_invalid(double x) {
    return std::isnan(x) || std::isinf(x);
}
}

class rbf {
public:
    rbf(points_t& _train_x, points_t& _train_y,
        unsigned _n_input, unsigned _n_hidden, unsigned _n_output,
        unsigned _epoch=100, unsigned _batch=5, double _min_loss=0.1, double _learning_rate=0.01, bool _verbose=true):

        train_x(_train_x), train_y(_train_y), n_input(_n_input), n_hidden(_n_hidden), n_output(_n_output),
        n_epoch(_epoch), n_batch(_batch), n_step(train_x.size()/n_batch), min_loss(_min_loss), learning_rate(_learning_rate), verbose(_verbose){

        weights = points_t(n_hidden, point_t(n_output, 0));
        bias = vec_t(n_output, 0);
        centers = points_t(n_hidden, point_t(n_input, 0));
        gammas = vec_t(n_hidden, 0);

        o1 = std::vector<double>(n_hidden);
        o2 = std::vector<double>(n_output);

        batch_dw = points_t(n_hidden, point_t(n_output, 0));
        batch_db = vec_t(n_output, 0);
        batch_dc = points_t(n_hidden, point_t(n_input, 0));
        batch_dr = vec_t(n_hidden, 0);

        init_weights();
    }
    void fit();
    void fit_kmeans();
    indexes_t predict(const points_t&);
    std::vector<vec_t> predict_regression(const points_t&);

private:
    void set_X(index_t x) {x_index = x;}
    void set_Y(index_t y) {y_index = y;}
    void forword_flow();
    void _forword_flow1();
    void _forword_flow1(const point_t&);
    void _forword_flow2();
    void backword_flow();
    void update_weights();
    double get_r();
    data_t gaussion(const point_t&, const point_t&, double);
    data_t dist(const point_t&, const point_t&);
    index_t argmax(const point_t&);
    void init_weights();

private:
    index_t x_index, y_index;
    points_t& train_x;
    points_t& train_y;
    unsigned n_input, n_hidden, n_output;
    points_t centers;
    vec_t gammas;
    std::vector<vec_t> weights;
    vec_t bias;
    vec_t o1, o2; //隐含层和输出层的输出结果

    std::vector<vec_t> batch_dw;
    vec_t batch_db;
    std::vector<vec_t> batch_dc;
    vec_t batch_dr;

public:
    double cur_loss;
    const unsigned n_epoch;
    const unsigned n_batch;
    const unsigned n_step;
    const double min_loss = 0.1;
    const double learning_rate = 0.01;
    const double L2 = 0;
    const bool verbose;
};

void
rbf::init_weights() {
    std::mt19937 gen;
    std::normal_distribution<double> normal(-0.1, 0.1);
    for(auto &_c:centers) for(auto &_e:_c) _e = normal(gen);
    for(auto &_e:gammas) _e = normal(gen);
    for(auto &_w:weights) for(auto &_e:_w) _e = normal(gen);
    for(auto &_e:bias) _e = normal(gen);

    std::cout << "init weights finished!" <<std::endl;
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
rbf::dist(const point_t &p1, const point_t &p2) {
    double _dist = 0.;
    for (index_t i=0; i<p1.size(); ++i) _dist += pow(p1[i]-p2[i], 2);
    return _dist;
}

double
rbf::gaussion(const point_t &p1, const point_t &p2, double r) {
    //return std::exp(-dist(p1, p2)/(2*(r*r)));
    return std::exp(r*dist(p1, p2)*0.5);
    double arg = r*dist(p1, p2)*0.5;
    if (arg>=15) return std::exp(15);
    if (arg<=-15) return std::exp(-15);
    double x = std::exp(arg);

    if (std::is_invalid(x)) {
        std::cout <<"gaussion:"<< r<<" " << dist(p1, p2);
        for (auto e:p1) std::cout << e << " ";
        std::cout << std::endl;
        for (auto e:p2) std::cout << e << " ";
        getchar();
    }
    return x;
}

double
rbf::get_r() {
    double max_dist = 0;
    for (index_t i=0; i<n_hidden; ++i) {
        for (index_t j=i+1; j<n_hidden; ++j) {
            max_dist = std::max(max_dist, dist(centers[i], centers[j]));
        }
    }
    return std::sqrt(max_dist/(2*n_hidden));
}

void
rbf::_forword_flow1(const point_t& p) {
    for (index_t i=0; i<n_hidden; ++i) {
        o1[i] = gaussion(p, centers[i], gammas[i]);
        //o1[i] = std::max(std::min(o1[i], 1e8), -1e8);
    }
}

void
rbf::_forword_flow1() {
    for (index_t i=0; i<n_hidden; ++i) {
        o1[i] = gaussion(train_x[x_index], centers[i], gammas[i]);
        //o1[i] = std::max(std::min(o1[i], 1e8), -1e8);
        if (std::is_invalid(o1[i])) {
            std::cout << "f1" << gammas[i] <<" " <<dist(train_x[x_index], centers[i])<< std::endl;
            for (auto e:train_x[x_index]) std::cout << e << " ";
            std::cout << std::endl;
            for (auto e:centers[i]) std::cout << e << " ";
            getchar();
        }
    }
}

void
rbf::_forword_flow2() {
    for (index_t i=0; i<n_output; ++i) {
        double sum = 0.;
        for (index_t j=0; j<n_hidden; ++j) {
            sum += o1[j]*weights[j][i];
            if (std::is_invalid(sum)) {
                std::cout << "f2_sum:"<<o1[j] << " " <<weights[j][i] <<" " << j;
                getchar();
            }
        }
        //sum = std::max(std::min(sum, 1e8), -1e8);
        o2[i] = sum + bias[i];
        if (std::is_invalid(o2[i])) {
            std::cout << "f2:" << sum << " " << bias[i] <<std::endl;
            getchar();
        }
        //o2[i] = sum;
    }
}

void
rbf::forword_flow() {
    _forword_flow1();
    _forword_flow2();
}

void
rbf::backword_flow() {
    //o2 bias sensitives
    //uesd to calculate "centers sensitives"

    for (index_t i=0; i<n_output; ++i) {
        batch_db[i] += o2[i] - train_y[y_index][i] + L2*bias[i];
        cur_loss += std::pow(o2[i] - train_y[y_index][i], 2);
        if (std::is_invalid(cur_loss)) {
            std::cout << "cur_loss is nan!" << std::endl;
            getchar();
            std::cout << o2[i] << "\t" << train_y[y_index][i] << "\t" << y_index << "\t" << i << std::endl;
            for (index_t j=0; j<n_hidden; ++j) {
                if (std::is_invalid(o1[j])) {
                    std::cout << "o1["<<j<<"]:"<<o1[j] << std::endl;
                    std::cout << "\ngamma["<<j<<"]:" << gammas[j]<<std::endl;
                    std::cout << gammas[0] << gammas[1] <<gammas[2] <<std::endl;
                    getchar();
                }
            }
        }
    }


    //weights sensitives
    for (index_t i=0; i<n_hidden; ++i) {
        for (index_t j=0; j<n_output; ++j) {
            batch_dw[i][j] += o1[i]*(o2[j] - train_y[y_index][j]) + L2*weights[i][j];
        }
    }

    for (index_t i=0; i<n_hidden; ++i) {
        double sum = 0.;
        for (index_t j=0; j<n_output; ++j) {
            sum += weights[i][j]*(o2[j] - train_y[y_index][j]);
            if (std::is_invalid(sum)) {
                std::cout << "sum:" << weights[i][j] << " " << o2[j] << " " << train_y[y_index][j];
                getchar();
            }
        }
        //centers sensitives
        //double tmp = sum*o1[i]/std::pow(gammas[i], 2);
        double tmp = sum*o1[i]*gammas[i];
        if (std::is_invalid(tmp)) {
            std::cout <<"batch_dc:"<< sum << " " << o1[i] << " " << gammas[i] << " " << tmp;
            getchar();
        }
        for (index_t m=0; m<n_input; ++m) {
            batch_dc[i][m] += tmp*(centers[i][m]-train_x[x_index][m]) + L2*centers[i][m];
        }
        //gammas sensitives
        //batch_dr[i] += sum*o1[i]*dist(centers[i], train_x[x_index])/std::pow(gammas[i], 3);
        batch_dr[i] += sum*o1[i]*dist(centers[i], train_x[x_index])*0.5 + L2*gammas[i];
    }
}

void
rbf::update_weights() {
    // update o2 bias
    for (index_t i=0; i<n_output; ++i) {
        bias[i] -= learning_rate*(batch_db[i]/n_batch);
        batch_db[i] = 0;
    }

    //update weights
    for (index_t i=0; i<n_hidden; ++i) {
        for (index_t j=0; j<n_output; ++j) {
            weights[i][j] -= learning_rate*(batch_dw[i][j]/n_batch);
            //weights[i][j] = std::max(std::min(weights[i][j], 1e8), -1e8);
            batch_dw[i][j] = 0;
        }
    }

    //update centers
    for (index_t i=0; i<n_hidden; ++i) {
        for (index_t j=0; j<n_input; ++j) {
            centers[i][j] -= learning_rate*(batch_dc[i][j]/n_batch);
            //centers[i][j] = std::max(std::min(centers[i][j], 1e8), -1e8);
            batch_dc[i][j] = 0;
        }
    }

    //update gammas
    for (index_t i=0; i<n_hidden; ++i) {
        gammas[i] -= learning_rate*(batch_dr[i]/n_batch);
        //gammas[i] = std::max(std::min(gammas[i], 1e8), -1e8);
        //gammas[i] += 1e-3;
        batch_dr[i] = 0;
    }

    //clear
    /*
    for(auto &_c:batch_dc) for(auto &_e:_c) _e = 0;
    for(auto &_e:batch_dr) _e = 0;
    for(auto &_w:batch_dw) for(auto &_e:_w) _e = 0;
    for(auto &_e:batch_db) _e = 0;
    */
}

void
rbf::fit_kmeans() {
    Kmeans k(train_x, n_hidden);
    k.cluster();
    centers = k.centers;
    double r = get_r();
    gammas = point_t(n_hidden, r);

    auto batch_o1 = points_t(n_batch, point_t(n_hidden, 0));
    double epoch_loss = 0;
    for (index_t e=0; e<n_epoch; ++e) {
        epoch_loss = 0;
        auto t1 = clock();

        for (index_t s=0; s<n_step; ++s) {
            //先保存中间结果
            for (index_t b=0; b<n_batch; ++b) {
                set_X(n_batch*s+b);
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
                for (index_t b=0; b<n_batch; ++b) {
                    o1 = batch_o1[b];
                    _forword_flow2();
                    for (index_t i=0; i<n_output; ++i) {
                        batch_db[i] += o2[i] - train_y[n_batch*s+b][i];
                        cur_loss += std::pow(o2[i] - train_y[n_batch*s+b][i], 2);
                        for (index_t j=0; j<n_hidden; ++j) {
                            batch_dw[j][i] += (o2[i] - train_y[n_batch*s+b][i])*o1[j];
                        }
                    }
                }
                for (index_t i=0; i<n_output; ++i) {
                    batch_db[i] /= n_batch;
                    for (index_t j=0; j<n_hidden; ++j) {
                        batch_dw[j][i] /= n_batch;
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
        std::cout << "epoch:" << e+1 << " loss:"<<epoch_loss/n_step <<" time:"<<(t2-t1)/double(CLOCKS_PER_SEC) << std::endl;
    }
}

void
rbf::fit() {
    double epoch_loss = 0;
    for (index_t e=0; e<n_epoch; ++e) {
        epoch_loss = 0;
        auto t1 = clock();
        nn::random_shuffle(train_x, train_x);
        for (index_t s=0; s<n_step; ++s) {
            int __itr = 0;
            cur_loss = min_loss+1;
            auto _t1 = clock();
            while (__itr++ < 500 && cur_loss>min_loss) {
                //auto tup1 = clock();
                update_weights();
                //auto tup2 = clock();
                //std::cout << "update time:" << (tup2-tup1) <<std::endl;
                cur_loss = 0;
                for (index_t b=0; b<n_batch; ++b) {
                    set_X(n_batch*s + b);
                    set_Y(n_batch*s + b);
                    //auto t11 = clock();
                    forword_flow();
                    //auto t12 = clock();
                    backword_flow();
                    //auto t13 = clock();
                    //std::cout << "forword time:" << (t12-t11) << " backword time:" << (t13-t12) <<std::endl;
                    //std::cout << "itr:" << __itr << " b:" << b << std::endl;
                }
            }
            epoch_loss += cur_loss/n_step;
            auto _t2 = clock();

            /*
            if (verbose) {
                std::cout << "epoch:"<<e+1 << " step:"<<s+1 << " step_loss:" << cur_loss <<" step_itr:"<< __itr <<
                             " step_time:"<<(_t2-_t1) <<std::endl;
            }*/
        }

        auto t2 = clock();
        if (verbose) {
            std::cout << "epoch:" << e+1 << " loss:"<<epoch_loss <<" time:"<<(t2-t1)/double(CLOCKS_PER_SEC) << std::endl;
        }
    }
}

indexes_t
rbf::predict(const points_t &test_x) {
    indexes_t res;
    auto _y = predict_regression(test_x);
    for (auto _o:_y) res.push_back(argmax(_o));
    return res;
}

std::vector<vec_t>
rbf::predict_regression(const points_t &test_x) {
    std::vector<point_t> res;
    for (index_t i=0; i<test_x.size(); ++i) {
        _forword_flow1(test_x[i]);
        _forword_flow2();
        res.push_back(o2);
    }
    return res;
}

#endif // RBF_H








