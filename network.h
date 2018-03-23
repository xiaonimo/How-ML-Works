#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include <random>
#include <assert.h>
#include "func.h"

class BP {
public:
    BP(){}
    BP(vector<int> l);

    void set_train_data(vector<vector<double>>, vector<vector<double>>);
    void set_test_data(vector<vector<double>>, vector<vector<double>>);
    void train(int _batch=10, int _max_itr_all=10, int _max_itr_batch=100,
               double _learning_rate=0.01, string _GD="SGD", string _activation="sigmoid",
               bool verbose=true);
    void predict(bool verbose=true);
    void data_normalization();

    void set_X(vector<double>);
    void set_Y(vector<double>);


public:
    vector<vector<double>> train_X, train_Y;
    vector<vector<double>> test_X, test_Y;
    vector<double> X, Y;

public:
    void forword_flow();
    void backword_flow();
    void update_weights_bias();
    void init_weights_bias();
    double active(double);
    double active_deri(double);

public:
    vector<double> weights;
    vector<double> bias;
    vector<double> d_weights;
    vector<double> d_bias;
    vector<double> vals;        //所有层都存储input和output
    
    vector<int> layers;         //存储每层神经元的个数
    vector<int> x_index;
    vector<int> y_index;
    vector<int> w_index;
    vector<int> b_index;

private:
    int get_x_start(int);
    int get_y_start(int);
    int get_w_start(int);
    int get_b_start(int);

    int n_layers;

    int batch;
    int max_itr_all;
    int max_itr_batch;
    double learning_rate;
    double min_loss;

    string GD;
    string activation = string("ReLu");
};

#endif // NETWORK_H
