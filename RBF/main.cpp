#include "rbf.hpp"
#include "mnist.hpp"
#include "kmeans.hpp"
#include <cstdio>

std::size_t argmax(const std::vector<double> &p) {
    std::size_t res = 0;
    double _max_v = p[res];

    for (index_t i=1; i<p.size(); ++i) {
        if (p[i] < _max_v) continue;
        _max_v = p[i];
        res = i;
    }
    return res;
}

void gen_data() {
    if (nullptr == std::freopen("data.csv", "w", stdout)) {
        throw std::invalid_argument("open file failed!");
    }
    std::mt19937 gen;
    std::uniform_real_distribution<double> urd(-10, 10);
    for (int i=0; i<100; ++i) {
        for (int j=0; j<100; ++j) {
            for (int k=0; k<10; ++k) {
                double a = urd(gen);
                double b = urd(gen);
                double c = urd(gen);
                printf("%lf,%lf,%lf,%lf\n", a, b, c, a*a + b*2 - 0.5*c);
            }
        }
    }
    std::fclose(stdout);
    //std::freopen("CON", "o", stdout);
    std::cout << "generate data finished!" << std::endl;
}

void read_data(std::vector<std::vector<data_t>>& X, std::vector<std::vector<data_t>>& Y, std::string filename) {
    if (X.size() != Y.size()) throw std::invalid_argument(" X and Y's size shoule be same!");
    unsigned num = X.size();
    unsigned dim = X[0].size();

    if(nullptr == std::freopen(filename.c_str(), "r", stdin)) {
        std::cout << "open " << filename << " failed!" <<std::endl;
    }
    data_t val = 0.;
    for (unsigned i=0; i<num; ++i) {
        for (unsigned j=0; j<dim+1; ++j) {
            scanf("%lf,", &val);
            if (j == dim) Y[i][0]=val;
            else X[i][j] = val;
        }
    }
    std::fclose(stdin);
    std::freopen("CON", "r", stdin);
    std::cout << "read " << filename << " finished!" << std::endl;
}

int main() {

    //Classification
    /*
    const int num = 10000;
    points_t X(num, point_t(784));
    points_t Y(num, point_t(10));
    read_mnist(X, Y, "train.csv");
    data_normalization(X);

    auto train_x = points_t(std::begin(X), std::begin(X)+int(num*0.85));
    auto train_y = points_t(std::begin(Y), std::begin(Y)+int(num*0.85));

    auto test_x = points_t(std::begin(X)+int(num*0.85), std::end(X));
    auto test_y = points_t(std::begin(Y)+int(num*0.85), std::end(Y));

    //Kmeans k(train_x, 1000);
    //k.cluster();

    rbf r(train_x, train_y, 784, 800, 10);
    r.train();
    auto pre = r.predict(test_x);

    int ca = 0;
    for (index_t i=0; i<test_y.size(); ++i) {
        ca += pre[i]==argmax(test_y[i]);
        std::cout << pre[i] << "/" << argmax(test_y[i]) <<std::endl;
    }
    std::cout << "accuracy:" << double(ca)/test_y.size() <<std::endl;
    */
    //gen_data();


    //Regression
    const int num = 10000;
    points_t X(num, point_t(3));
    points_t Y(num, point_t(1));
    read_data(X, Y, "data.csv");

    auto train_x = points_t(std::begin(X), std::begin(X)+int(num*0.85));
    auto train_y = points_t(std::begin(Y), std::begin(Y)+int(num*0.85));

    auto test_x = points_t(std::begin(X)+int(num*0.85), std::end(X));
    auto test_y = points_t(std::begin(Y)+int(num*0.85), std::end(Y));

    rbf r(train_x, train_y, 3, 10, 1);
    r.train();
    auto pre = r.predict_regression(test_x);
    for (index_t i=0; i<test_x.size(); ++i) {
        std::cout << pre[i][0] << "/" << test_y[i][0] <<std::endl;
    }

    return 0;
}
