#ifndef RBF_HPP
#define RBF_HPP

#include <array>
#include <random>
#include "datatype.hpp"
#include "kmeans.hpp"

class rbf {
public:
    rbf(const points_t& _train_x, const points_t& _train_y, unsigned _n_input, unsigned _n_hidden, unsigned _n_output, bool _verbose=true):
       train_x(_train_x), train_y(_train_y), n_input(_n_input), n_hidden(_n_hidden), n_output(_n_output), verbose(_verbose){
        //r = std::vector<double>(n_hidden);
        weights = std::vector<std::vector<double>>(n_hidden, std::vector<double>(n_output));
        bias = std::vector<double>(n_output);
        o1 = std::vector<double>(n_hidden);
        o2 = std::vector<double>(n_output);
    }
    void train();
    std::vector<std::size_t> predict(const points_t&);

private:
    void set_X(index_t x) {x_index = x;}
    void set_Y(index_t y) {y_index = y;}
    void forword_flow();
    void backword_flow();
    void update_loss();
    data_t get_r();
    data_t gaussion(const point_t&, const point_t&);
    data_t get_dist(const point_t&, const point_t&);
    std::size_t argmax(const std::vector<double>&);
    void init_weights_bias();

private:
    std::size_t x_index, y_index;
    const points_t& train_x;
    const points_t& train_y;
    unsigned n_input, n_hidden, n_output;
    points_t c;
    double r;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::vector<double> o1, o2; //隐含层和输出层的输出结果
    double cur_loss;
    bool verbose;
};

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
rbf::gaussion(const point_t &p1, const point_t &p2) {
    return std::exp(r*get_dist(p1, p2));
}

void
rbf::forword_flow() {
    for (index_t i=0; i<n_hidden; ++i) {
        o1[i] = gaussion(train_x[i], c[i]);
    }
    for (index_t i=0; i<n_output; ++i) {
        double sum = 0.;
        for (index_t j=0; j<n_hidden; ++j) {
            sum += o1[j]*weights[j][i];
        }
        o2[i] = sum + bias[i];
    }
}

void
rbf::backword_flow() {
    // update bias
    for (index_t i=0; i<n_output; ++i) {
        bias[i] -= o2[i]-train_y[y_index][i];
    }
    for (index_t i=0; i<n_hidden; ++i) {
        for (index_t j=0; j<n_output; ++j) {
            weights[i][j] -= (o2[j]-train_y[y_index][j])*o1[i];
        }
    }
}

void
rbf::train() {
    Kmeans k(points_t(std::begin(train_x), std::begin(train_x)+4000), n_hidden);
    c = k.centers;
    r = get_r();

    for (index_t i=0; i<train_x.size(); ++i) {
        set_X(i); set_Y(i);
        int itr = 0;
        while (itr++ < 10) {
            forword_flow();
            backword_flow();
            update_loss();
        }
        if (verbose) std::cout << "itr:" << i << " loss:" << cur_loss << std::endl;
    }
}

std::vector<std::size_t>
rbf::predict(const points_t &test_x) {
    std::vector<std::size_t> res;
    for (index_t i=0; i<test_x.size(); ++i) {
        set_X(i);
        forword_flow();
        res.push_back(argmax(o2));
    }
    return res;
}

#endif // RBF_H








