#ifndef MLP_HPP
#define MLP_HPP

#include <ctime>
#include <vector>
#include <string>
#include <random>
#include <assert.h>
#include <cfloat>
#include <fstream>
#include <sstream>
#include <initializer_list>

#include "datatype.hpp"
#include "func.hpp"


class MLP {
public:
    MLP(points_t &_train_x, points_t &_train_y, points_t &_test_x, points_t &_test_y,
       std::initializer_list<unsigned> l, unsigned _n_epoch=20, unsigned _n_batch=5,
       double _learning_rate=0.05, double _min_loss=0.01, double _momentum=0.9,
       std::string _activation=std::string("sigmoid"), double _L1=0., double _L2=0.001, bool _verbose=true):

        train_X(_train_x), train_Y(_train_y), test_X(_test_x), test_Y(_test_y),
        layers(l), n_layers(l.size()), n_epoch(_n_epoch), n_batch(_n_batch), n_step(train_X.size()/n_batch),
        learning_rate(_learning_rate), min_loss(_min_loss), momentum(_momentum), L1(_L1), L2(_L2),
        activation(_activation), verbose(_verbose)
    {
        init();
        init_weights_bias();
        init_sigmoid_table();
    }

    void fit();
    index_t predict(const point_t&);
    vec_t predict_prob(const point_t&);
    std::vector<index_t> predict(const points_t &);
    std::vector<vec_t> predict_prob(const points_t&);
    void save_model(std::string);
    void load_model(std::string);

public:
    points_t &train_X, &train_Y, &test_X, &test_Y;
    std::vector<unsigned> layers;             //存储每层神经元的个数
    const unsigned n_layers;
    const unsigned n_epoch;
    const unsigned n_batch;
    const unsigned n_step;
    const double learning_rate;
    const double min_loss;
    const double momentum;
    const double L1, L2;
    const std::string activation;
    bool verbose;

private:
    void init();
    void init_weights_bias();       //参数初始化
    void forword_flow();            //前向传播
    void backword_flow();           //反向传播
    void _backword_flow1();       //输出层到上一层的梯度
    void _backword_flow2();        //其他层的梯度
    void update_weights_bias();     //根据梯度，更新参数

    double active(double);          //激活函数
    double active_deri(double);     //激活函数导数

private:
    std::vector<double> weights, dw, batch_dw, last_vw;         //权重值
    std::vector<double> bias, db, batch_db, last_vb;
    std::vector<double> vals;            //所有层都存储input和output

    unsigned get_layer_neurals(int);     //获得某一层的节点数

    std::vector<unsigned> x_index;            //根据某一层获得x,y,w,b的初始index
    std::vector<unsigned> y_index;
    std::vector<unsigned> w_index;
    std::vector<unsigned> b_index;
    unsigned get_x_start(int);
    unsigned get_y_start(int);
    unsigned get_w_start(int);
    unsigned get_b_start(int);

    double cur_loss;
    point_t Y;
    void set_X(const point_t&);
    void set_Y(const point_t&);

    std::vector<double> sigmoid_table;
    void init_sigmoid_table();
};

void
MLP::init(){
    //初始化各种index，方便每层快速定位
    x_index.assign(n_layers, 0);
    y_index.assign(n_layers, 0);
    w_index.assign(n_layers, 0);
    b_index.assign(n_layers, 0);
    for (index_t i=0; i<n_layers; ++i) {
        x_index[i] = i?x_index[i-1]+2*layers[i-1]:0;
        y_index[i] = i?y_index[i-1]+layers[i-1]+layers[i]:layers[0];
        if (i == 1 || i==0) continue;
        w_index[i] = w_index[i-1]+layers[i-2]*layers[i-1];
        b_index[i] = b_index[i-1]+layers[i-1];
    }

    //计算参数数量
    int sum_weights=0;
    int sum_bias = 0;
    for (index_t i=1; i<n_layers;++i) {
        sum_weights += layers[i-1]*layers[i];
        sum_bias += layers[i];
    }

    //参数初始化
    weights.assign(sum_weights, 0.);
    bias.assign(sum_bias, 0.);
    dw.assign(sum_weights, 0.);
    db.assign(sum_bias, 0.);
    last_vw.assign(sum_weights, 0.);
    last_vb.assign(sum_bias, 0.);
    batch_dw.assign(sum_weights, 0);
    batch_db.assign(sum_bias, 0);
    vals.assign(2*(sum_bias+layers[0]), 0);

    sigmoid_table.assign(4000, 0.);
}

void
MLP::init_weights_bias() {
    //根据正太分布初始化参数
    std::mt19937 gen;
    std::normal_distribution<double> normal(-0.001, 0.001);

    for (index_t i=0; i<weights.size(); ++i) weights[i]=normal(gen);
    for (index_t i=0; i<bias.size(); ++i) bias[i]=normal(gen);
}

void
MLP::set_X(const point_t& x) {
    //X = x;
    unsigned _x_start = get_x_start(0);
    unsigned _y_start = get_y_start(0);
    for (int i=0; i<int(x.size()); ++i) {
        vals[_x_start+i] = x[i];
        vals[_y_start+i] = x[i];
    }
}

void
MLP::set_Y(const point_t &y) {
    Y = y;
}

void
MLP::forword_flow() {
    //除输入层，将所有vals值清零
    for (index_t i=2*layers[0]; i<vals.size(); ++i) vals[i]=0;
    //每一层操作
    for (index_t cur_layer=0; cur_layer<n_layers-1; ++cur_layer) {
        //获取各个数据的起点
        const unsigned _x_start_next = get_x_start(cur_layer+1);
        const unsigned _y_start_cur  = get_y_start(cur_layer);
        const unsigned _w_start_next = get_w_start(cur_layer+1);
        //各层的每个神经元操作
        for (index_t cur_cell=0; cur_cell<layers[cur_layer]; ++cur_cell) {
            //每个神经元对应 不同的权重连接
            for (index_t cur_w=0; cur_w<layers[cur_layer+1]; ++cur_w) {
                //定位到下一个输入层的起点
                vals[_x_start_next+cur_w] += vals[_y_start_cur+cur_cell]
                        *weights[_w_start_next+cur_cell*layers[cur_layer+1]+cur_w];
            }
        }
        //加上bias
        const unsigned _b_start_next = get_b_start(1);
        for (index_t i=0; i<layers[cur_layer+1]; ++i) {
            vals[_x_start_next+i] += bias[_b_start_next+i];
        }
        //通过激活函数
        const unsigned _y_start_next = get_y_start(cur_layer+1);
        for (index_t i=0; i<layers[cur_layer+1]; ++i) {
            vals[_y_start_next+i] = active(vals[_x_start_next+i]);
        }
    }
}

void
MLP::_backword_flow1() {
    //输出层与隐含层之间的参数更新
    const unsigned hidden1_layer_size = get_layer_neurals(-2);
    const unsigned output_layer_size  = get_layer_neurals(-1);

    const unsigned _w_start_output    = get_w_start(-1);//最后一层（输出层）对应的权重连接
    const unsigned _b_start_output    = get_b_start(-1);
    const unsigned _x_start_output    = get_x_start(-1);
    const unsigned _y_start_output    = get_y_start(-1);
    const unsigned _y_start_hidden1   = get_y_start(-2);
    //先计算d_bias
    for (index_t output_cell=0; output_cell<output_layer_size; ++output_cell) {
        //L2 loss
        db[_b_start_output+output_cell] = (vals[_y_start_output+output_cell]-Y[output_cell])
                *active_deri(vals[_x_start_output+output_cell]);
        cur_loss += std::pow(vals[_y_start_output+output_cell]-Y[output_cell], 2)/n_batch;

        /*cross-entropy loss
        double y = Y[output_cell];
        double _y = vals[_y_start_output+output_cell];
        d_bias[_b_start_output+output_cell] = _y-y;
        */
    }

    //更新隐含层每个神经元与所有输出层神经元的权重
    for (index_t hidden1_cell=0; hidden1_cell<hidden1_layer_size; ++hidden1_cell) {
        //隐含层单个神经元与每个输出层神经元相连
        for (index_t output_cell=0; output_cell<output_layer_size; ++output_cell) {
            const unsigned __w_index = _w_start_output + hidden1_cell*output_layer_size + output_cell;
            dw[__w_index] = db[_b_start_output+output_cell]*vals[_y_start_hidden1+hidden1_cell]+
                    (L2/weights.size())*weights[__w_index];
        }
    }
}

void
MLP::_backword_flow2() {
    //循环遍历每一层
    for (index_t cur_layer=n_layers-2; cur_layer>0; --cur_layer) {
        const unsigned cur_layer_size  = get_layer_neurals(cur_layer);
        const unsigned pre_layer_size  = get_layer_neurals(cur_layer-1);
        const unsigned next_layer_size = get_layer_neurals(cur_layer+1);
        //上一层每个神经元
        const unsigned _b_start_cur  = get_b_start(cur_layer);
        const unsigned _b_start_next = get_b_start(cur_layer+1);
        const unsigned _w_start_cur  = get_w_start(cur_layer);
        const unsigned _w_start_next = get_w_start(cur_layer+1);
        const unsigned _x_start_cur  = get_x_start(cur_layer);
        const unsigned _y_start_pre  = get_y_start(cur_layer-1);
        //先计算d_bias
        for (index_t cur_hidden_cell=0; cur_hidden_cell<cur_layer_size; ++cur_hidden_cell) {
            double sum = 0.;
            for (index_t next_hidden_cell=0; next_hidden_cell<next_layer_size; ++next_hidden_cell) {
                sum += db[_b_start_next+next_hidden_cell]
                        *weights[_w_start_next+cur_hidden_cell*next_layer_size+next_hidden_cell];
            }
            db[_b_start_cur+cur_hidden_cell] = sum*active_deri(vals[_x_start_cur+cur_hidden_cell]);
        }

        //计算d_weights
        for (index_t pre_hidden_cell=0; pre_hidden_cell<pre_layer_size; ++pre_hidden_cell) {
            //与当前层每个神经元连接的权重
            for (index_t cur_hidden_cell=0; cur_hidden_cell<cur_layer_size; ++cur_hidden_cell) {
                dw[_w_start_cur+pre_hidden_cell*cur_layer_size+cur_hidden_cell]
                        =db[_b_start_cur+cur_hidden_cell]*vals[_y_start_pre+pre_hidden_cell];
            }
        }
    }
}

void
MLP::backword_flow() {
    _backword_flow1();
    _backword_flow2();
}

double
MLP::active(double x) {
    if (activation == std::string("sigmoid")) {
        if (x>=20) return 0.999999999999;
        else if (x <= -20) return 0.0000000000001;
        else return sigmoid_table[int((x+20)*100)];
    }
    else if (activation == std::string("relu")) {
        return x>0?x:0;
    }
    else {
        return x;
        std::cout << "active error" << std::endl;
    }
}

double
MLP::active_deri(double x) {
    if (activation == std::string("sigmoid")) {
        x = active(x);
        return x*(1-x);
    }
    else if (activation == std::string("relu")) {
        return x>0?1:0;
    }
    else {
        return x;
        std::cout << "active_deri error" << std::endl;
    }
    return 0.;
}

unsigned
MLP::get_x_start(int cur_layer) {
    return x_index[(cur_layer+n_layers)%n_layers];
}

unsigned
MLP::get_y_start(int cur_layer) {
    return y_index[(cur_layer+n_layers)%n_layers];
}

unsigned
MLP::get_w_start(int cur_layer) {
    assert(cur_layer !=0);
    return w_index[(cur_layer+n_layers)%n_layers];
}

unsigned
MLP::get_b_start(int cur_layer) {
    assert(cur_layer!=0);
    return b_index[(cur_layer+n_layers)%n_layers];
}

unsigned
MLP::get_layer_neurals(int cur_layer) {
    return layers[(cur_layer+n_layers)%n_layers];
}

void
MLP::fit() {
    unsigned w_num = unsigned(weights.size());
    unsigned b_num = unsigned(bias.size());
    for (index_t e=0; e<n_epoch; ++e) {
        double epoch_loss = 0.;
        auto t1 = clock();
        nn::random_shuffle(train_X, train_Y);
        for (index_t s=0; s<n_step; ++s) {
            int _itr = 0;
            cur_loss = min_loss + 1;
            //auto _t1 = clock();
            while (cur_loss > min_loss && _itr++<100) {
                cur_loss = 0.;
                update_weights_bias();
                batch_dw.assign(w_num, 0);
                batch_db.assign(b_num, 0);
                for (index_t b=0; b<n_batch; ++b) {
                    set_X(train_X[s*n_batch+b]);
                    set_Y(train_Y[s*n_batch+b]);
                    forword_flow();
                    backword_flow();
                    for (index_t _w=0; _w<w_num; ++_w) batch_dw[_w] += dw[_w]/n_batch;
                    for (index_t _b=0; _b<b_num; ++_b) batch_db[_b] += db[_b]/n_batch;
                }
            }
            //auto _t2 = clock();
            //std::cout << "step:" << s+1 << " loss:" << cur_loss << " time:" << _t2-_t1 <<std::endl;
            epoch_loss += cur_loss/n_batch;
        }
        auto epoch_res = predict(test_X);
        unsigned ca = 0;
        for (index_t i=0; i<test_X.size(); ++i) {
            ca += nn::argmax(test_Y[i])==epoch_res[i];
        }
        auto t2 = clock();
        if (verbose) {
            std::cout << "epoch:" << e+1 << "\tloss:" << epoch_loss << "accuracy:" << double(ca)/test_X.size() << "\ttime:" << t2-t1 << std::endl;
        }
    }
}

void
MLP::update_weights_bias() {
    if (fabs(momentum)<1e-5) {
        for (index_t i=0; i<weights.size(); ++i) weights[i] -= learning_rate*batch_dw[i];
        for (index_t i=0; i<bias.size(); ++i) bias[i] -= learning_rate*batch_db[i];
    }
    //const double g = 0.9;
    for (index_t i=0; i<weights.size(); ++i) {
        //last_v_weights[i] = momentum*last_v_weights[i]-(learning_rate*d_weights[i]+(weight_decay/(int)weights.size())*weights[i]);
        last_vw[i] = momentum*last_vw[i] - learning_rate*batch_dw[i];
        weights[i] += last_vw[i];
    }
    for (index_t i=0; i<bias.size(); ++i) {
        last_vb[i] = momentum*last_vb[i] - learning_rate*batch_db[i];
        bias[i] += last_vb[i];
    }
}

index_t
MLP::predict(const point_t& p) {
    return nn::argmax(predict_prob(p));
}

vec_t
MLP::predict_prob(const point_t& p) {
    set_X(p);
    forword_flow();
    const unsigned _y_start_output = get_y_start(-1);
    const unsigned output_layer_size = get_layer_neurals(-1);
    auto res = vec_t(output_layer_size);
    for (index_t i=0; i<output_layer_size; ++i) {
        res[i] = vals[_y_start_output+i];
    }
    return res;
}

std::vector<index_t>
MLP::predict(const points_t &test_x) {
    std::vector<index_t> res;
    for (index_t i=0; i<test_x.size(); ++i) {
        res.push_back(predict(test_x[i]));
    }
    return res;
}

std::vector<vec_t>
MLP::predict_prob(const points_t&test_x) {
    std::vector<vec_t> res;
    for (index_t i=0; i<test_x.size(); ++i) {
        res.push_back(predict_prob(test_x[i]));
    }
    return res;
}

void
MLP::init_sigmoid_table() {
    double __s = 40.0/4000.0;
    for (int i=0; i<4000; ++i) {
        sigmoid_table[i] = 1.0/(1+exp(20-i*__s));
    }
    std::cout << "init sigmoid table finished" << std::endl;
}

void
MLP::save_model(std::string filename) {
    std::ofstream fw(filename);
    if (!fw.is_open()) {
        std::cout << filename << " not opened!" << std::endl;
        exit(0);
    }
    //保存网络结构
    for (index_t i=0; i<n_layers; ++i) {
        fw << layers[i];
        if (i!=n_layers-1) fw <<",";
        else fw << std::endl;
    }
    //保存激活函数
    fw << activation << std::endl;
    //保存权重
    for (int i=0; i<(int)weights.size(); ++i) {
        fw << weights[i];
        if (i!=(int)weights.size()-1) fw<<",";
        else fw<< std::endl;
    }
    //保存bias
    for (int i=0; i<(int)bias.size(); ++i) {
        fw << bias[i];
        if (i!=(int)bias.size()-1) fw<<",";
        else fw<< std::endl;
    }
    fw.close();
}
/*
void
MLP::load_model(string filename) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cout <<filename << " not opened!" <<endl;
        exit(0);
    }
    //读取layer信息
    vector<int> layer;
    string line;
    getline(fin, line);
    istringstream lin(line);
    string val;
    while (getline(lin, val, ',')) {
        int _n;
        stringstream ss(val);
        ss>>_n;
        layer.push_back(_n);
    }

    //创建对象
    points_t &rf = points_t();
    MLP m(rf, rf, rf, rf, layer);

    //读取激活函数信息
    getline(fin, line);
    m.activation = line;

    //读取weights信息
    getline(fin, line);
    istringstream lin2(line);
    int i=0;
    while (getline(lin2, val, ',')) {
        double _w;
        stringstream ss(val);
        ss >> _w;
        m.weights[i++] = _w;
    }

    //读取bias信息
    getline(fin, line);
    istringstream lin3(line);
    i=0;
    while (getline(lin3, val, ',')) {
        double _w;
        stringstream ss(val);
        ss >> _w;
        m.bias[i++] = _w;
    }
    *this = m;
}
*/

#endif // MLP_HPP
