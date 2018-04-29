#ifndef RBF_HPP
#define RBF_HPP

#include <iomanip>
#include <random>
#include "datatype.hpp"
#include "kmeans.hpp"

class rbf {
public:
    rbf(const points_t& _train_x, const points_t& _train_y, unsigned _n_input, unsigned _n_hidden, unsigned _n_output, bool _verbose=true):
       train_x(_train_x), train_y(_train_y), n_input(_n_input), n_hidden(_n_hidden), n_output(_n_output), verbose(_verbose){
        rs = std::vector<double>(n_hidden);
        weights = std::vector<std::vector<double>>(n_hidden, std::vector<double>(n_output));
        bias = std::vector<double>(n_output);
        o1 = std::vector<double>(n_hidden);
        o2 = std::vector<double>(n_output);
    }
    void train();
    std::vector<std::size_t> predict(const points_t&);
    std::vector<point_t> predict_regression(const points_t&);

private:
    void set_X(index_t x) {x_index = x;}
    void set_Y(index_t y) {y_index = y;}
    void forword_flow();
    void _forword_flow();
    void backword_flow();
    void update_loss();
    data_t get_r();
    void cal_r();
    data_t gaussion(const point_t&, const point_t&);
    data_t gaussion(const point_t&, const point_t&, double);
    data_t get_dist(const point_t&, const point_t&);
    std::size_t argmax(const std::vector<double>&);
    void init_weights_bias();
    void init_exp_table();

private:
    std::size_t x_index, y_index;
    const points_t& train_x;
    const points_t& train_y;
    unsigned n_input, n_hidden, n_output;
    points_t c;
    double r;
    std::vector<double> rs;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::vector<double> o1, o2, delta; //隐含层和输出层的输出结果
    double cur_loss;
    bool verbose;
    double learning_rate = 0.01;
    double min_loss = 0.0001;
    std::vector<double> exp_table;
};

void
rbf::init_exp_table() {

}

void
rbf::init_weights_bias() {
    std::mt19937 gen;
    std::normal_distribution<double> normal(0, 0.0001);
    for(auto &_w:weights) for(auto &_e:_w) _e = normal(gen);
    for(auto &_e:bias) _e = normal(gen);
}

void
rbf::update_loss() {
    cur_loss = get_dist(train_y[y_index], o2);
}

std::size_t
rbf::argmax(const std::vector<double> &p) {
    std::size_t res = 0;
    double _max_v = p[res];

    for (index_t i=1; i<p.size(); ++i) {
        if (p[i] < _max_v) continue;
        _max_v = p[i];
        res = i;
    }
    return res;
}

double
rbf::get_dist(const point_t &p1, const point_t &p2) {
    double _dist = 0.;
    for (index_t i=0; i<p1.size(); ++i) _dist += pow(p1[i]-p2[i], 2);
    return _dist;
}

void
rbf::cal_r() {
    rs[0] = get_dist(c[0], c[1]);
    rs[n_hidden-1] = get_dist(c[n_hidden-1], c[n_hidden-2]);
    for (index_t i=1; i<n_hidden-1; ++i) {
        rs[i] = std::max(get_dist(c[i], c[i-1]), get_dist(c[i], c[i+1]));
    }
    for (index_t i=0; i<n_hidden; ++i) {
        rs[i] = -1 / (2 * std::pow(rs[i]/std::sqrt(c.size()*2), 2));
    }
}

double
rbf::get_r() {
    double _max_dist = std::numeric_limits<double>::min();
    for (index_t i=0; i<c.size(); ++i) {
        for (index_t j=i+1; j<c.size(); ++j) {
            double _dist = get_dist(c[i], c[j]);
            if (_dist > _max_dist) _max_dist = _dist;
        }
    }
    double _r = _max_dist/std::sqrt(2*c.size());
    return -1/(2*std::pow(_r, 2));
}

double
rbf::gaussion(const point_t &p1, const point_t &p2, double _r) {
    return std::exp(_r*get_dist(p1, p2));
}

double
rbf::gaussion(const point_t &p1, const point_t &p2) {
    return std::exp(r*get_dist(p1, p2));
}

void
rbf::_forword_flow() {
    for (index_t i=0; i<n_output; ++i) {
        double sum = 0.;
        for (index_t j=0; j<n_hidden; ++j) {
            sum += o1[j]*weights[j][i];
        }
        //o2[i] = sum + bias[i];
        o2[i] = sum;
    }
}

void
rbf::forword_flow() {
    for (index_t i=0; i<n_hidden; ++i) {
        o1[i] = gaussion(train_x[x_index], c[i], r);
    }
    _forword_flow();
}

void
rbf::backword_flow() {
    /* update bias
    for (index_t i=0; i<n_output; ++i) {
        bias[i] -= learning_rate*(o2[i]-train_y[y_index][i]);
    }*/

    //update weights
    for (index_t i=0; i<n_hidden; ++i) {
        for (index_t j=0; j<n_output; ++j) {
            weights[i][j] -= learning_rate*(o2[j]-train_y[y_index][j])*o1[i];
        }
    }
}

void
rbf::train() {
    //cluster to calculate centers and r;
    Kmeans k(train_x, n_hidden);
    k.cluster();
    c = k.centers;
    r = get_r();
    cal_r();

    // SGD
    /*
    int itr_all = 0;
    while (itr_all++ < 100) {
        auto t1 = clock();
        for (index_t i=0; i<train_x.size(); ++i) {
            set_X(i); set_Y(i);
            int itr = 0;
            forword_flow();
            update_loss();
            while (itr++ < 500 && cur_loss > min_loss) {
                backword_flow();
                _forword_flow();
                update_loss();
            }
        }
        auto t2 = clock();
        if(verbose) {
            std::cout << "itr:"<<itr_all+1 << " loss:"<< cur_loss <<" time:" << (t2-t1)/double(CLOCKS_PER_SEC) <<std::endl;
        }
    }*/

    //BGD
    const index_t epoch = 1000;
    const index_t batch = 10;
    const index_t step = train_x.size()/batch;
    auto tmp = std::vector<point_t>(batch, point_t(n_hidden, 0));
    for (index_t e=0; e<epoch; ++e) {
        auto t1 = clock();
        for (index_t s=0; s<step; ++s) {
            //先保存中间结果
            for (index_t b=0; b<batch; ++b) {
                set_X(batch*s+b); //set_Y(batch*s+b);
                forword_flow();
                tmp[b] = o1;
            }
            int __itr = 0;
            cur_loss = min_loss+1;
            while (__itr++ < 500 && cur_loss>min_loss) {
                cur_loss = 0;
                delta = std::vector<double>(n_output, 0);

                for (index_t b=0; b<batch; ++b) {
                    o1 = tmp[b];
                    _forword_flow();
                    for (index_t i=0; i<n_output; ++i) {
                        delta[i] += (o2[i] - train_y[batch*s+b][i])/batch;
                        cur_loss += std::pow(delta[i], 2)/batch;
                    }
                }
                for (index_t i=0; i<n_hidden; ++i) {
                    for (index_t j=0; j<n_output; ++j) {
                        weights[i][j] -= learning_rate*(delta[j])*o1[i];
                    }
                }
            }
        }
        auto t2 = clock();
        std::cout << "epoch:" << e+1 << " loss:"<<cur_loss <<" time:"<<(t2-t1)/double(CLOCKS_PER_SEC) << std::endl;
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








