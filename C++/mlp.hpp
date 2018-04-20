#ifndef BP_NETWORK_H
#define BP_NETWORK_H

#include <vector>
#include <string>
#include <random>
#include <cfloat>
#include <cstdlib> // freopen
#include <iomanip>
#include <fstream>
#include <sstream>
#include <initializer_list> //for constructor parameters
#include "mat.hpp"


class MLP {
public://functions
    MLP(std::initializer_list<unsigned int> layers, std::string activation="sigmoid", std::string solver="adam",
        std::string loss="mse", double min_loss=0.1, double alpha=0.0001,
        unsigned int batch_size=10, double learning_rate=0.001, double momentum=0.9, double validation_fraction=0.1,
        unsigned int max_itr_all=10, unsigned int max_itr_batch=100, bool verbose=false, bool early_stop=false);
    void fit(const std::vector<Mat>&, const std::vector<Mat>&);
    std::size_t predict(const Mat&);
    std::vector<std::size_t> predict(const std::vector<Mat>&);
    std::vector<double> predict_proba(const Mat &);
    std::vector<std::vector<double>> predict_proba(const std::vector<Mat>&);

public://network's parameters
    std::vector<Mat> weights, d_weights, last_v_weights;
    std::vector<Mat> bias, d_bias, last_v_bias;

    std::vector<Mat> io_data;//input and ouput of every layer except input_layer

public://data
    //std::vector<Mat> train_X;           //训练样本
    //std::vector<Mat> train_Y;           //训练样本的label
    //std::vector<Mat> test_X;            //测试样本
    //std::vector<Mat> test_Y;            //测试样本的label
    Mat cur_X;                          //当前训练样本
    Mat cur_Y;                          //当前训练样本的label

public://hyper-parameters
    std::vector<unsigned int> layers;       //每一层的信息
    unsigned int n_layers;
    std::string activation;
    std::string solver;
    std::string loss;
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

public:
    void construct_network();
    void _init(Mat&, int);              //矩阵初始化
    void init_network();                //初始化w,b

    void set_X(const Mat&);             //设置当前需要训练的样本
    void set_Y(const Mat&);             //设置当前需要训练的样本的label

    //void show_output();               //显示某一次迭代的结果
    void forward_flow();                //前向传播
    void backword_flow();               //反向传播
    void update_weights_bias();         //根据梯度，更新参数
    void update_loss();
    void train_itr();                   //优化一个样本，迭代次数控制迭代结束
    void train_loss();                  //优化一个样本，loss阈值控制迭代结束

    std::size_t argmax(Mat);            //vector数组中最大元素的下标
    std::size_t argmax(std::vector<double>);
    void create_sigmoid_table();        //创建一个sigmoid表，加快计算速度
    double sigmoid(double);            //对单独的值进行sigmoid计算
    double active(const double);
    Mat active(Mat&);
    double active_deri(const double);
    Mat active_deri(Mat&);

private:
    std::vector<double> sigmoid_table;       //sigmoid表
    unsigned int cur_itr_batch;         //当前batch的迭代次数
    double cur_loss;                    //当前loss
};

void
MLP::train_loss() {
    cur_loss = DBL_MAX_10_EXP;
    unsigned int __itr = 0;
    while (cur_loss > min_loss && __itr++<max_itr_batch) {
        forward_flow();
        update_loss();
        backword_flow();
        update_weights_bias();
        //std::cout << __itr <<":" <<cur_loss <<std::endl;
    }
}

void
MLP::update_loss() {
    cur_loss = 0.;
    //L2 loss
    if (loss == std::string("mse")) {
        cur_loss = (io_data.back()-cur_Y).square().sum()*0.5;
    }

    //Cross-Entropy loss
    if (loss == std::string("ec")) {
        for (std::size_t i=0; i<layers.back(); ++i) {
            double y = cur_Y[0][i], _y=io_data.back()[0][i];
            cur_loss += -y*std::log(_y) - (1-y)*std::log(1-_y);
        }
    }
}

void
MLP::update_weights_bias() {
    for (std::size_t i=0; i<weights.size(); ++i) {
        last_v_weights[i] = last_v_weights[i]*momentum-(d_weights[i]*learning_rate);
        weights[i] = weights[i] + last_v_weights[i];

        last_v_bias[i] = last_v_bias[i]*momentum - d_bias[i]*learning_rate;
        bias[i] = bias[i] + last_v_bias[i];
    }
}

Mat
MLP::active_deri(Mat& x) {
    Mat res(x.rows, x.cols);
    for (std::size_t r=0; r<x.rows; ++r) {
        for (std::size_t c=0; c<x.cols; ++c) res[r][c] = active(x[r][c]);
    }
    return res;
}

double
MLP::active_deri(const double x) {
    if (activation == std::string("sigmoid")) {
        double y = active(x);
        return y*(1-y);
    }
    else if (activation == std::string("relu")) return x>0?1:0;
    else {
        return x;
        std::cout << "active_deri error" << std::endl;
    }
}

Mat
MLP::active(Mat &x) {
    Mat res(x.rows, x.cols);
    for (std::size_t r=0; r<x.rows; ++r) {
        for (std::size_t c=0; c<x.cols; ++c) res[r][c] = active(x[r][c]);
    }
    return res;
}

double
MLP::active(const double x) {
    if (activation == std::string("sigmoid")) {
        if (x>=20) return 0.999999999999;
        else if (x <= -20) return 0.0000000000001;
        else return sigmoid_table[int((x+20)*100)];
    }
    if (activation == std::string("relu")) return x>0?x:0;
    else {
        return x;
        std::cout << "active error" << std::endl;
    }
}

void
MLP::create_sigmoid_table() {
    double __s = 40.0/4000.0;
    for (int i=0; i<4000; ++i) sigmoid_table.push_back(1.0/(1+exp(20-i*__s)));
}

std::size_t
MLP::argmax(Mat x) {
    Mat::T _max_val = DBL_MIN;
    std::size_t res = -1;
    for (std::size_t c=0; c<x.cols; ++c) {
        if (x[0][c] < _max_val) continue;
        _max_val = x[0][c];
        res = c;
    }
    return res;
}

std::size_t
MLP::argmax(std::vector<double> x) {
    std::size_t res = -1;
    double _max_val =  DBL_MIN;
    for (std::size_t i=0; i<x.size(); ++i) {
        if (_max_val > x[i]) continue;
        res = i;
        _max_val = x[i];
    }
    return res;
}

MLP::MLP(std::initializer_list<unsigned int> layers, std::string activation, std::string solver,
         std::string loss, double min_loss, double alpha, unsigned int batch_size, double learning_rate,
         double momentum, double validation_fraction, unsigned int max_itr_all,
         unsigned int max_itr_batch, bool verbose, bool early_stop) {
    this->layers = layers;
    this->n_layers = layers.size();
    this->activation = activation;
    this->solver = solver;
    this->loss = loss;
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
    if (this->activation == std::string("sigmoid")) create_sigmoid_table();
    construct_network();
}

void
MLP::fit(const std::vector<Mat> &x, const std::vector<Mat> &y) {
    if (x.size() != y.size()) throw std::invalid_argument("x.size != y.size");
    //validation
    const unsigned int data_sz = x.size();
    const unsigned int k_cross = (unsigned int)(1/validation_fraction);
    const unsigned int k_sz = data_sz * validation_fraction;

    //k-cross validation
    for (unsigned _k=0; _k<k_cross; ++_k) {
        unsigned _lower_bound = _k*k_sz;
        unsigned _upper_bound = (_k+1)*k_sz;
        //train
        for (unsigned __itr_all=0; __itr_all<max_itr_all; ++__itr_all) {
            for (unsigned _index=0; _index<data_sz; ++_index) {
                if (_index>=_lower_bound && _index <_upper_bound) continue;
                set_X(x[_index]); set_Y(y[_index]);
                train_loss();
            }
            std::cout << "cross-validation:" << _k+1 << " itr_all:" << __itr_all+1 <<
                         " cur_loss:" <<cur_loss <<std::endl;
        }
        //predict
        unsigned int correct_pred = 0;
        unsigned int n_predict=0;
        for (unsigned _index=_lower_bound; _index<_upper_bound && _index<data_sz; ++_index) {
            //set_X(x[_index]); set_Y(y[_index]);
            std::size_t _p = predict(x[_index]);
            correct_pred += argmax(y[_index])==_p;
            n_predict++;
        }
        double _accy_k = (double)correct_pred/n_predict;
        std::cout << _k+1 << "th validation, accuracy=" << std::setprecision(10) << _accy_k <<std::endl;
    }
}

std::size_t
MLP::predict(const Mat &x) {
    return argmax(predict_proba(x));
}

std::vector<double>
MLP::predict_proba(const Mat &x) {
    set_X(x);
    forward_flow();

    std::vector<double> res;
    Mat _output = io_data.back();
    for (unsigned r=0; r<_output.rows; ++r) {
        for (unsigned c=0; c<_output.cols; ++c) res.push_back(_output[r][c]);
    }
    return res;
}

std::vector<std::size_t>
MLP::predict(const std::vector<Mat> &x) {
    std::vector<std::size_t> res;
    for (auto _x:x) res.push_back(predict(_x));
    return res;
}

std::vector<std::vector<double>>
MLP::predict_proba(const std::vector<Mat> &x) {
    std::vector<std::vector<double>> res;
    for (auto _x:x) res.push_back(predict_proba(_x));
    return res;
}

void
MLP::set_X(const Mat &x) {cur_X = x;}

void
MLP::set_Y(const Mat &y) {cur_Y = y;}

void
MLP::construct_network() {
    for (std::size_t i=1; i<n_layers; ++i) {
        weights.push_back(Mat(layers[i-1], layers[i]));
        bias.push_back(Mat(1, layers[i]));
        d_weights.push_back(Mat(layers[i-1], layers[i]));
        d_bias.push_back(Mat(1, layers[i]));
        last_v_weights.push_back(Mat(layers[i-1], layers[i]));
        last_v_bias.push_back(Mat(1, layers[i]));
        io_data.push_back(Mat(1, layers[i]));
        io_data.push_back(Mat(1, layers[i]));
    }
    init_network();
}

void
MLP::_init(Mat &x, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0, 0.0001);
    for (std::size_t r=0; r<x.rows; ++r) {
        for (std::size_t c=0; c<x.cols; ++c) x[r][c] = nd(gen);
    }
}

void
MLP::init_network() {
    for (std::size_t i=0; i<weights.size(); ++i) _init(weights[i], i);
}

void
MLP::forward_flow() {
    for (std::size_t i=0; i<n_layers-1; ++i) {
        if (i==0) io_data[i*2] = cur_X*weights[i] + bias[i];
        else io_data[i*2] = io_data[i*2-1]*weights[i] + bias[i];
        io_data[i*2+1] = active(io_data[i*2]);
    }
}

void
MLP::backword_flow() {
    //output_layer -> hidden_layer
    //std::cout << "io_data.back():"<<std::endl;
    //io_data.back().shape();
    //std::cout << "cur_Y.shape:"<<std::endl;
    //cur_Y.shape();
    //d_bias.back().shape();
    if (loss == std::string("mse")) d_bias.back() = (io_data.back()-cur_Y).mul(active_deri(io_data[io_data.size()-2]));
    if (loss == std::string("ce")) d_bias.back() = io_data.back()-cur_Y;
    //io_data[io_data.size()-3].shape(); d_bias.back().shape();
    d_weights.back() = io_data[io_data.size()-3].inverse()*d_bias.back();

    //hidden_layer -> input_layer
    for (int i=weights.size()-2; i>=0; --i) {
        //weights[i+1].shape();d_bias[i+1].shape();io_data[i*2].shape();
        d_bias[i] = (d_bias[i+1]*weights[i+1].inverse()).mul(active_deri(io_data[i*2]));
        if (i==0) d_weights[i] = cur_X.inverse()*d_bias[i];
        else d_weights[i] = io_data[i*2-1].inverse()*d_bias[i];
    }
}


void read_train_mnist(std::vector<Mat>& X, std::vector<Mat>& Y, std::string filename) {
    int num = X.size();

    freopen(filename.c_str(), "r", stdin);
    double val = 0.;
    for (int i=0; i<num; ++i) {
        for (int j=0; j<784+1; ++j) {
            scanf("%lf,", &val);
            if (j == 0) Y[i][0][int(val)] = 1;
            else X[i][0][j-1] = val;
        }
    }
    fclose(stdin);
    std::cout << "read mnist data finished" << std::endl;
}
#endif
