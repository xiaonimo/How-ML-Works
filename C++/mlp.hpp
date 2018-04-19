#ifndef BP_NETWORK_H
#define BP_NETWORK_H

#include <vector>
#include <string>
#include <random>
#include <cfloat>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <initializer_list> //for constructor parameters
#include "mat.hpp"


class MLP {
public://functions
    MLP(std::initializer_list<unsigned int> layers, std::string activation="sigmoid", std::string solver='adam',
        double min_loss=0.1, double alpha=0.0001,
        unsigned int batch_size=10, double learning_rate=0.001, double momentum=0.9, double validation_fraction=0.1,
        unsigned int max_itr_all=10, unsigned int max_itr_batch=100, bool verbose=false, bool early_stop=false);
    void fit(const std::vector<Mat>&, const std::vector<Mat>&);
    void predict(const std::vector<Mat>&);
    void predict_proba(const std::vector<Mat>&);

public://network's parameters
    std::vector<Mat> weights;
    std::vector<Mat> bias;

    std::vector<Mat> d_weights;
    std::vector<Mat> d_bias;

public://data
    std::vector<Mat> train_X;           //训练样本
    std::vector<Mat> train_Y;           //训练样本的label
    std::vector<Mat> test_X;            //测试样本
    std::vector<Mat> test_Y;            //测试样本的label
    Mat cur_X;                          //当前训练样本
    Mat cur_Y;                          //当前训练样本的label

public://hyper-parameters
    std::initializer_list<unsigned int> layers;       //每一层的信息
    unsigned int n_layers;
    std::string activation;
    std::string solver;
    double alpha;//for L2
    unsigned int batch_size;
    double learning_rate;
    double momentum;
    double validation_fraction;
    unsigned int max_itr_all;           //最外层的迭代次数
    unsigned int max_itr_batch;         //每个batch的最多迭代次数
    double min_loss;                    //loss阈值
    bool verbose;
    bool early_stop;

private:
    void construct_network();
    void _init(Mat&, int);              //矩阵初始化
    void init_network();                //初始化w,b

    void set_X(Mat);                    //设置当前需要训练的样本
    void set_Y(Mat);                    //设置当前需要训练的样本的label

    //void show_output();               //显示某一次迭代的结果
    void forward_flow();                //前向传播
    void backword_flow();               //反向传播
    void update_weights();              //根据梯度，更新参数
    void train_itr();                   //优化一个样本，迭代次数控制迭代结束
    void train_loss();                  //优化一个样本，loss阈值控制迭代结束

    std::size_t argmax(Mat);            //vector数组中最大元素的下标
    void create_sigmoid_table();        //创建一个sigmoid表，加快计算速度
    Mat sigmoid(Mat);                   //对矩阵每个元素进行sigmoid计算
    double _sigmoid(double);            //对单独的值进行sigmoid计算

private:
    vector<double> sigmoid_table;       //sigmoid表
    unsigned int cur_itr_batch;         //当前batch的迭代次数
    double cur_loss;                    //当前loss
};

MLP::MLP(std::initializer_list<unsigned int> layers, std::string activation, std::string solver,
         double min_loss, double alpha, unsigned int batch_size, double learning_rate,
         double momentum, double validation_fraction, unsigned int max_itr_all,
         unsigned int max_itr_batch, bool verbose, bool early_stop) {
    this->layers = layers;
    this->activation = activation;
    this->solver = solver;
    this->min_loss = min_loss;
    this->alpha = alpha;
    this->batch_size = batch_size;
    this->learning_rate = learning_rate;
    this->momentum = momentum;
    this->validation_fraction = validation_fraction;
    this->max_itr_all = max_itr_all;
    this->max_itr_batch = max_itr_batch;
    this->verbose = verbose;
    this->early_stop = early_stop;
    if (activation == string("sigmoid")) create_sigmoid_table();
}

void MLP::fit(const std::vector<Mat> &x, const std::vector<Mat> &y) {
    if (x.size() != y.size()) throw std::invalid_argument("x.size != y.size");
    //validation
    const unsigned int data_sz = x.size();
    const unsigned int k_cross = unsigned int(1/validation_fraction);
    const unsigned int k_sz = data_sz * validation_fraction;

    //k-cross validation
    for (unsigned _k=0; _k<k_cross; ++_k) {
        unsigned _lower_bound = _k*k_sz;
        unsigned _upper_bound = (_k+1)*k_sz;
        //train
        for (unsigned _index=0; _index<data_sz; ++_index) {
            if (_index>=_lower_bound && _index <_upper_bound) continue;
            set_X(x[_index]); set_Y(y[_index]);
            train_loss();
        }
        //predict
        for (unsigned _index=0; _index<data_sz; ++_index) {

            if (_index>=_lower_bound && _index <_upper_bound) {

            }
        }
    }
}


#endif
