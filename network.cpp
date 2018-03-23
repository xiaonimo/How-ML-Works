#include "network.h"

BP::BP(vector<int> l){
    assert(int(l.size())>=3);
    layers = l;
    n_layers = layers.size();

    //初始化各种index，方便每层快速定位
    x_index = vector<int>(n_layers, 0);
    y_index = vector<int>(n_layers, 0);
    w_index = vector<int>(n_layers, 0);
    b_index = vector<int>(n_layers, 0);
    for (int i=0; i<n_layers; ++i) {
        x_index[i] = i?x_index[i-1]+2*layers[i-1]:0;
        y_index[i] = i?y_index[i-1]+layers[i-1]+layers[i]:layers[0];
        if (i == 0) continue;
        w_index[i] = i?w_index[i-1]+layers[i]*layers[i-1]:0;
        b_index[i] = i?b_index[i-1]+layers[i]:0;
    }

    //计算参数数量
    int sum_weights=0;
    int sum_bias = 0;
    for (int i=1; i<n_layers;++i) {
        sum_weights += layers[i-1]*layers[i];
        sum_bias += layers[i];
    }

    //参数初始化
    weights = vector<double>(sum_weights, 0.);
    bias = vector<double>(sum_bias, 0.);
    d_weights = vector<double>(sum_weights, 0.);
    bias = vector<double>(sum_bias, 0.);
    vals = vector<double>(2*(sum_bias+layers[0]), 0);

    init_weights_bias();
}

void BP::init_weights_bias() {
    mt19937 gen;
    normal_distribution<double> normal(0, 0.001);

    for (int i=0; i<int(weights.size()); ++i) weights[i]=normal(gen);
    for (int i=0; i<int(bias.size()); ++i) bias[i]=normal(gen);
}

void BP::set_X(vector<double> x) {
    X = x;
    int _x_start = get_x_start(0);
    int _y_start = get_y_start(0);
    for (int i=0; i<int(x.size()); ++i) {
        vals[_x_start+i] = x[i];
        vals[_y_start+i] = x[i];
    }
}

void BP::set_Y(vector<double> y) {
    Y = y;
}

void BP::forword_flow() {
    //除输入层，将所有vals值清零
    for (int i=2*layers[0]; i<int(vals.size()); ++i) vals[i]=0;
    //每一层操作
    for (int cur_layer=0; cur_layer<n_layers-1; ++cur_layer) {
        //获取各个数据的起点
        int _x_start_next = get_x_start(cur_layer+1);
        int _y_start_cur = get_y_start(cur_layer);
        int _w_start_next = get_w_start(cur_layer+1);
        //各层的每个神经元操作
        for (int cur_cell=0; cur_cell<layers[cur_layer]; ++cur_cell) {
            //每个神经元对应 不同的权重连接
            for (int cur_w=0; cur_w<layers[cur_layer+1]; ++cur_w) {
                //定位到下一个输入层的起点
                vals[_x_start_next+cur_w] += vals[_y_start_cur+cur_cell]
                        *weights[_w_start_next+cur_cell*layers[cur_layer+1]+cur_w];
            }
        }
        //加上bias
        int _b_start_next = get_b_start(1);
        for (int i=0; i<layers[cur_layer+1]; ++i) {
            vals[_x_start_next+i] += bias[_b_start_next+i];
        }
        //通过激活函数
        int _y_start_next = get_y_start(cur_layer+1);
        for (int i=0; i<layers[cur_layer+1]; ++i) {
            vals[_y_start_next+i] = active(vals[_x_start_next+i]);
        }
    }
}

 void BP::backword_flow() {
    //输出层与隐含层之间的参数更新
    int last_hidden_layer_size = layers[n_layers-2];
    int output_layer_size = layers.back();

    int _w_index = (int)weights.size()-last_hidden_layer_size*output_layer_size;
    int _net_index = (int)vals.size()-2*layers.back();
    int _output_index = (int)vals.size()-layers.back();
    int _x_index = _net_index - last_hidden_layer_size;
    int _b_index = (int)bias.size()-layers.back();
    //更新隐含层每个神经元与所有输出层神经元的权重
    for (int hidden_cell=0; hidden_cell<last_hidden_layer_size; ++hidden_cell) {
        //隐含层单个神经元与每个输出层神经元相连
        for (int output_cell=0; output_cell<output_layer_size; ++output_cell) {
            int __w_index = _w_index + hidden_cell*output_layer_size + output_cell;
            int __b_index = _b_index + output_cell;
            d_weights[__w_index] = (vals[_output_index+output_cell]-Y[output_cell])
                    *active_deri(vals[_net_index+output_cell])*vals[_x_index+hidden_cell];
            d_bias[__b_index] = (vals[_output_index+output_cell]-Y[output_cell])
                    *active_deri(vals[_net_index+output_cell]);
        }
    }

    //倒数第一隐含层的参数更新
    _w_index = w_index[n_layers-3];
    int _w_index_next = w_index[n_layers-2];
    _b_index = b_index[n_layers-3];
    int _y_index_next = y_index[n_layers-2];
    int _x_index_next = x_index[n_layers-2];
    //_net_index =


    //倒数第2隐含层，逐个更新神经元与下一层所有神经元的连接
    for (int hidden_cell =0; hidden_cell<layers[n_layers-3]; ++hidden_cell) {
        //逐个遍历下一层所有神经元
        for (int next_hidden_cell=0; next_hidden_cell<layers[n_layers-2]; ++next_hidden_cell) {
            //当前神经元与输出层链接，计算sum((y-Y)*f'(net)*w)
            double sum = 0.;
            for (int output_cell=0; output_cell<layers[n_layers-1]; ++output_cell) {
                sum += (vals[_y_index_next+output_cell]-Y[output_cell])
                        *active(vals[_x_index_next+output_cell])
                        *weights[_w_index_next+next_hidden_cell*layers.back()+output_cell];
            }
            //获得d_b
            //d_bias[_b_index+next_hidden_cell] = sum*active(1);
            //d_weights[_w_index + 1] = d_bias[_b_index+next_hidden_cell] * y;
        }
    }

    //其他隐含层之间的参数更新,根据上一层的d_b
    for (int cur_layer = n_layers-2; cur_layer>=1; ++cur_layer) {

    }
}

double BP::active(double x) {
    if (activation == string("sigmoid")) return 1.0/(1+exp(-x));
    if (activation == string("ReLu")) return x>0?x:0;
    else {
        return x;
        cout << "active error" << endl;
    }
}

double BP::active_deri(double x) {
    if (activation == string("sigmoid")) {
        x = active(x);
        return x*(1-x);
    }
    if (activation == string("ReLu")) return x>0?1:0;
    else {
        return x;
        cout << "active_deri error" << endl;
    }
}

int BP::get_x_start(int cur_layer) {
    assert(cur_layer>=0 && cur_layer <n_layers);
    return x_index[(cur_layer+n_layers)%n_layers];
}

int BP::get_y_start(int cur_layer) {
    assert(cur_layer>=0 && cur_layer<n_layers);
    return y_index[(cur_layer+n_layers)%n_layers];
}

int BP::get_w_start(int cur_layer) {
    assert(cur_layer>0 && cur_layer<n_layers);
    return w_index[(cur_layer+n_layers)%n_layers];
}

int BP::get_b_start(int cur_layer) {
    assert(cur_layer>0 && cur_layer<n_layers);
    return b_index[(cur_layer+n_layers)%n_layers];
}
