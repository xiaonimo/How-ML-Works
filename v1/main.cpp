#include "network.h"
#include "func.h"
#include <ctime>


int main() {
    vector<vector<double>> X(40000, vector<double>(784, 0));
    vector<vector<double>> Y(40000, vector<double>(10, 0));
    read_mnist(X, Y, "train.csv");
    data_normal(X);

    /*
    auto t1 = clock();
    BP a(vector<int>{784, 100, 10});
    //a.load_model("model1.txt");
    a.set_train_data(X, Y, 0.9);
    a.train(10, 10, 1000, 0.01, 0.01, "SGD", "sigmoid");
    a.predict();
    a.save_model("model_36000.txt");
    auto t2 = clock();
    cout << (t2-t1)/1000.;
    */
    BP a;
    a.load_model("model_36000.txt");
    a.set_train_data(X, Y, 0);
    a.predict();

}
