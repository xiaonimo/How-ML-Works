#ifndef FUNC_HPP
#define FUNC_HPP

#include <cstdio>
#include <vector>
#include <array>
#include <string>
#include <random>
#include <iostream>
#include <stdexcept>
#include "datatype.hpp"

namespace nn {

void
read_mnist(points_t& X, points_t& Y, std::string filename) {
    if (X.size() != Y.size()) throw std::invalid_argument(" X and Y's size shoule be same!");
    unsigned num = X.size();

    if(!std::freopen(filename.c_str(), "r", stdin)) {
        throw std::invalid_argument("open " + filename + " failed!");
    }
    data_t val = 0.;
    for (index_t i=0; i<num; ++i) {
        for (index_t j=0; j<784+1; ++j) {
            scanf("%lf,", &val);
            if (j == 0) Y[i][int(val)] = 1;
            else X[i][j-1] = val;
        }
    }
    std::fclose(stdin);
    std::freopen("CON", "r", stdin);
    std::cout << "read " << filename << " finished!" << std::endl;
}

void
read_mnist(std::vector<std::array<std::array<double, 28>, 28>> &X, std::vector<std::array<double, 10>> &Y, std::string filename) {
    if (X.size() != Y.size()) throw std::invalid_argument(" X and Y's size shoule be same!");
    unsigned num = X.size();

    if(!std::freopen(filename.c_str(), "r", stdin)) {
        throw std::invalid_argument("open " + filename + " failed!");
    }
    data_t val = 0.;
    for (index_t i=0; i<num; ++i) {
        for (index_t j=0; j<784+1; ++j) {
            scanf("%lf,", &val);
            if (j == 0) Y[i][int(val)] = 1;
            else X[i][(j-1)/28][(j-1)%28] = double(val)/255.;
        }
    }
    std::fclose(stdin);
    std::freopen("CON", "r", stdin);
    std::cout << "read " << filename << " finished!" << std::endl;
}

void
data_normalization1(points_t& X) {
    for (auto &r:X) for (auto &e:r) e/=255.;
}

void
data_normalization2(points_t &X) {
    std::vector<data_t> _max(X[0]), _min(X[0]);
    for (unsigned r=0; r<X.size(); ++r) {
        for (unsigned c=0; c<X[0].size(); ++c)  {
            _max[c] = std::max(_max[c], X[r][c]);
            _min[c] = std::min(_min[c], X[r][c]);
        }
    }
    for (unsigned r=0; r<X.size(); ++r) {
        for (unsigned c=0; c<X[0].size(); ++c)  {
            X[r][c] = (X[r][c]-_min[c])/(_max[c]-_min[c]);
        }
    }
}

template<typename T1, typename T2>
void random_shuffle(std::vector<T1>& X, std::vector<T2>& Y) {
    std::mt19937 gen;
    std::uniform_int_distribution<int> uid(0, X.size()-1);
    for (index_t i=0; i<X.size(); ++i) {
        auto _i = uid(gen);
        std::swap(X[i], X[_i]);
        std::swap(Y[i], Y[_i]);
    }
}

template<typename T>
index_t argmax(const std::vector<T> &p) {
    std::size_t res = 0;
    T _max_v = p[res];

    for (index_t i=1; i<p.size(); ++i) {
        if (p[i] < _max_v) continue;
        _max_v = p[i];
        res = i;
    }
    return res;
}

template<typename T>
index_t argmax(std::array<T, 10> x) {
    T _tmp = x[0];
    index_t res = 0;
    for (index_t i=1; i<10; ++i) {
        if (x[i] <= _tmp) continue;
        _tmp = x[i];
        res = i;
    }
    return res;
}

void
gen_data() {
    std::string filename("data_mul.csv");
    if (!std::freopen(filename.c_str(), "w", stdout)) {
        throw std::invalid_argument("open " + filename + " failed!");
    }
    std::mt19937 gen;
    std::uniform_real_distribution<double> urd(-10, 10);
    /*
    for (int i=0; i<100; ++i) {
        for (int j=0; j<100; ++j) {
            for (int k=0; k<10; ++k) {
                double a = urd(gen);
                double b = urd(gen);
                double c = urd(gen);
                printf("%lf,%lf,%lf,%lf\n", a, b, c, a+b+c);
            }
        }
    }*/
    for (int i=0; i<20000; ++i) {
        double a = urd(gen);
        printf("%lf,%lf\n", a, a*a + std::cos(a));
    }
    std::fclose(stdout);
    //std::freopen("CON", "o", stdout);
    std::cout << "generate data finished!" << std::endl;
}

void
read_data(points_t& X, points_t& Y, std::string filename) {
    if (X.size() != Y.size()) throw std::invalid_argument(" X and Y's size shoule be same!");
    unsigned num = X.size();
    unsigned dim = X[0].size();

    if(nullptr == std::freopen(filename.c_str(), "r", stdin)) {
        throw std::invalid_argument("open " + filename + " failed!");
    }
    data_t val = 0.;
    for (index_t i=0; i<num; ++i) {
        for (index_t j=0; j<dim+1; ++j) {
            scanf("%lf,", &val);
            if (j == dim) Y[i][0]=val;
            else X[i][j] = val;
        }
    }
    std::fclose(stdin);
    std::freopen("CON", "r", stdin);
    std::cout << "read " << filename << " finished!" << std::endl;
}

template<typename T, unsigned t1, unsigned t2>
void padding(std::array<std::array<T, t1>, t1>& x, std::array<std::array<T, t2>, t2>& y) {
    if (t1<t2 || (t1-t2)%2) throw std::invalid_argument("");
    T e=0;
    unsigned sz = (t1-t2)/2;
    for (unsigned i=0; i<t1; ++i) {
        for (unsigned j=0; j<t1; ++j) {
            if (i<sz || i>=sz+t2) e=0;
            else if (j<sz || j>=sz+t2) e=0;
            else e=y[i-sz][j-sz];
            x[i][j] = e;
        }
    }
}

} //namespace
#endif // FUNC_HPP
