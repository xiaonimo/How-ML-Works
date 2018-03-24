#include "network.h"
#include "func.h"
#include <ctime>
double f(double x) {
    return 1.0/(1+exp(-x));
}

int main() {
    BP a(vector<int>{784, 100 ,10});

    vector<vector<double>> X(1000, vector<double>(784, 0));
    vector<vector<double>> Y(1000, vector<double>(10, 0));
    read_mnist(X, Y, "train.csv");
    data_normal(X);
    a.set_train_data(X, Y, 0.8);
    a.train(10, 3, 1000, 0.01, 0.1, "SGD", "sigmoid");
    a.predict();
}
