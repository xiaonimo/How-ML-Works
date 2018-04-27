#ifndef MNIST_HPP
#define MNIST_HPP

#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include "datatype.hpp"

void read_mnist(std::vector<std::vector<data_t>>& X, std::vector<std::vector<data_t>>& Y, std::string filename) {
    if (X.size() != Y.size()) throw std::invalid_argument(" X and Y's size shoule be same!");
    unsigned num = X.size();

    if(nullptr == std::freopen(filename.c_str(), "r", stdin)) {
        std::cout << "open " << filename << " failed!" <<std::endl;
    }
    data_t val = 0.;
    for (unsigned i=0; i<num; ++i) {
        for (unsigned j=0; j<784+1; ++j) {
            scanf("%lf,", &val);
            if (j == 0) Y[i][int(val)] = 1;
            else X[i][j-1] = val;
        }
    }
    std::fclose(stdin);
    std::freopen("CON", "r", stdin);
    std::cout << "read mnist data finished" << std::endl;
}

void data_normalization(std::vector<std::vector<data_t>>& X) {
    for (unsigned r=0; r<X.size(); ++r) {
        for (unsigned c=0; c<X[0].size(); ++c)  {
            X[r][c]=X[r][c]?1:0;
        }
    }
}


#endif // MNIST_HPP
