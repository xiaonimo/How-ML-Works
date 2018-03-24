#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include <random>
#include <assert.h>
#include <cfloat>
#include "func.h"

class BP {
public:
    BP(vector<int> l);

    void set_train_data(vector<vector<double>>, vector<vector<double>>, double r=0.8);
    void train(int _batch=10, int _max_itr_all=3, int _max_itr_batch=1000,
               double _learning_rate=0.01, double _min_loss=0.1, string _GD="SGD", string _activation="sigmoid",
               bool _verbose=true);
    void predict(bool _verbose=true);

public:
    vector<vector<double>> train_X, train_Y;
    vector<vector<double>> test_X, test_Y;

private:
    void forword_flow();            //前向传播
    void backword_flow();           //反向传播
    void bf_output_hidden1();       //输出层到上一层的梯度
    void bf_hidden1_input();        //其他层的梯度
    void get_loss();                //计算当前loss

    void update_weights_bias();     //根据梯度，更新参数
    void init_weights_bias();       //参数初始化
    double active(double);          //激活函数
    double active_deri(double);     //激活函数导数

private:
    vector<double> weights;         //权重值
    vector<double> bias;
    vector<double> d_weights;       //w梯度
    vector<double> d_bias;          //b梯度
    vector<double> vals;            //所有层都存储input和output
    
    vector<int> layers;             //存储每层神经元的个数
    int n_layers;
    int get_layer_neurals(int);     //获得某一层的节点数

    vector<int> x_index;            //根据某一层获得x,y,w,b的初始index
    vector<int> y_index;
    vector<int> w_index;
    vector<int> b_index;
    int get_x_start(int);
    int get_y_start(int);
    int get_w_start(int);
    int get_b_start(int);

    vector<double> X, Y;            //当前传播的XY
    void set_X(vector<double>);
    void set_Y(vector<double>);

private:
    int batch;                      //各种超参数
    int max_itr_all;
    int max_itr_batch;
    double learning_rate;
    double min_loss;
    double cur_loss;
    bool verbose;

    string GD;
    string activation;
};

#endif // NETWORK_H
