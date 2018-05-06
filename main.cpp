#include "RBF.hpp"
#include "func.hpp"
#include "Kmeans.hpp"
#include "LR.hpp"
#include "mLR.hpp"
#include "LeNet.hpp"
#include "mlp.hpp"
#include <cstdio>

int main() {
    /*MLP
    const unsigned num = 40000;
    points_t X(num, point_t(784));
    points_t Y(num, point_t(10));
    nn::read_mnist(X, Y, "train.csv");
    nn::data_normalization1(X);

    auto train_x = points_t(std::begin(X), std::begin(X)+int(num*0.9));
    auto train_y = points_t(std::begin(Y), std::begin(Y)+int(num*0.9));

    auto test_x = points_t(std::begin(X)+int(num*0.9), std::end(X));
    auto test_y = points_t(std::begin(Y)+int(num*0.9), std::end(Y));

    MLP m(train_x, train_y, test_x, test_y, {784, 100, 10});
    m.fit();
    */

    //LeNet
    const unsigned num = 4000;

    std::vector<std::array<std::array<double, 28>, 28>> x28(num);
    std::vector<std::array<std::array<double, 32>, 32>> x32(num);
    std::vector<std::array<double, 10>> y(num);

    nn::read_mnist(x28, y, "train.csv");
    std::cout << "read data finished!" << std::endl;
    for (index_t i=0; i<4000; ++i) nn::padding(x32[i], x28[i]);
    std::cout << "padding data finished!" << std::endl;

    auto train_x = std::vector<std::array<std::array<double, 32>, 32>>(begin(x32), begin(x32)+num*0.9);
    auto test_x  = std::vector<std::array<std::array<double, 32>, 32>>(begin(x32)+num*0.9, end(x32));

    auto train_y = std::vector<std::array<double ,10>>(begin(y), begin(y)+num*0.9);
    auto test_y  = std::vector<std::array<double ,10>>(begin(y)+num*0.9, end(y));

    auto t1 = clock();
    LeNet a(train_x, train_y, test_x, test_y);
    a.fit();
    auto t2 = clock();
    auto res = a.predict(test_x);
    int correct_answer = 0;
    for (unsigned i=0; i<test_y.size(); ++i) {
        index_t _pre = nn::argmax(test_y[i]);
        correct_answer += res[i]==_pre;
        std::cout << _pre << "/" << res[i] << std::endl;
    }
    auto t3 = clock();
    std::cout << "accuracy:" << double(correct_answer)/test_y.size() << std::endl;
    std::cout << "train time:" << t2-t1 << "\tpredict time:" << t3-t2 << std::endl;
    //

    /*LR
    const int num = 40000;
    points_t X(num, point_t(784));
    points_t _Y(num, point_t(10));
    nn::read_mnist(X, _Y, "train.csv");
    nn::data_normalization1(X);
    std::vector<index_t> Y;
    for (auto e:_Y) {
        Y.push_back(nn::argmax(e));
    }

    auto train_x = points_t(std::begin(X), std::begin(X)+int(num*0.9));
    auto train_y = std::vector<index_t>(std::begin(Y), std::begin(Y)+int(num*0.9));

    auto test_x = points_t(std::begin(X)+int(num*0.9), std::end(X));
    auto test_y = std::vector<index_t>(std::begin(Y)+int(num*0.9), std::end(Y));

    MLR mlr(train_x, train_y, 10);
    mlr.fit();
    auto res = mlr.predict_prob(test_x);
    unsigned ca = 0;
    for (index_t i=0; i<test_x.size(); ++i) {
        //std::cout << res[i] << "\t" << test_y[i] <<std::endl;
        ca += res[i]==test_y[i];
    }
    std::cout << "accuracy:" <<double(ca)/test_x.size() << std::endl;
    */

    /* Regression- rbf
    const int num = 10000;
    points_t X(num, point_t(3));
    points_t Y(num, point_t(1));
    nn::read_data(X, Y, "data.csv");
    nn::data_normalization2(X);
    nn::data_normalization2(Y);

    auto train_x = points_t(std::begin(X), std::begin(X)+int(num*0.85));
    auto train_y = points_t(std::begin(Y), std::begin(Y)+int(num*0.85));

    auto test_x = points_t(std::begin(X)+int(num*0.85), std::end(X));
    auto test_y = points_t(std::begin(Y)+int(num*0.85), std::end(Y));

    rbf r(train_x, train_y, {3, 100, 1}, 100, 5, 0.0001, 0.01);
    r.fit();
    auto pre = r.predict_regression(test_x);
    for (index_t i=0; i<test_x.size(); ++i) {
        std::cout << pre[i][0] << "/" << test_y[i][0] <<std::endl;
    }
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
    auto pre1 = r.predict_regression(train_x);
    for (index_t i=0; i<test_x.size(); ++i) {
        std::cout << pre1[i][0] << "/" << train_y[i][0] <<std::endl;
    }
    */

    return 0;
}
