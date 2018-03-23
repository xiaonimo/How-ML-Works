#include "network.h"
#include "func.h"

int main() {
    BP a(vector<int>{784,100,10});

    /*
    print(a.x_index);
    print(a.y_index);
    print(a.w_index);
    print(a.b_index);
    print(a.weights);
    print(a.bias);
    print(a.vals);
    */

    vector<vector<double>> X(2000, vector<double>(784, 0));
    vector<vector<double>> Y(2000, vector<double>(10, 0));
    read_mnist(X, Y, "train.csv");
    data_normal(X);
    a.set_train_data(X, Y);
    a.train();
    a.predict();
}
