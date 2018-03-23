#include "network.h"
#include "func.h"

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
        if (i == 1 || i==0) continue;
        w_index[i] = w_index[i-1]+layers[i-2]*layers[i-1];
        b_index[i] = b_index[i-1]+layers[i];
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
    d_bias = vector<double>(sum_bias, 0.);
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

void BP::bf_output_hidden1() {
    //输出层与隐含层之间的参数更新
    int hidden1_layer_size = get_layer_neurals(-2);
    int output_layer_size = get_layer_neurals(-1);

    int _w_start_output = get_w_start(-1);//最后一层（输出层）对应的权重连接
    int _b_start_output = get_b_start(-1);
    int _x_start_output = get_x_start(-1);
    int _y_start_output = get_y_start(-1);
    int _y_start_hidden1 = get_y_start(-2);
    //更新隐含层每个神经元与所有输出层神经元的权重
    for (int hidden1_cell=0; hidden1_cell<hidden1_layer_size; ++hidden1_cell) {
        //隐含层单个神经元与每个输出层神经元相连
        for (int output_cell=0; output_cell<output_layer_size; ++output_cell) {
            int __w_index = _w_start_output + hidden1_cell*output_layer_size + output_cell;
            int __b_index = _b_start_output + output_cell;
            d_weights[__w_index] = (vals[_y_start_output+output_cell]-Y[output_cell])
                    *active_deri(vals[_x_start_output+output_cell])*vals[_y_start_hidden1+hidden1_cell];
            d_bias[__b_index] = (vals[_y_start_output+output_cell]-Y[output_cell])
                    *active_deri(vals[_x_start_output+output_cell]);
        }
    }
}

void BP::bf_hidden1_hidden2() {
    int hidden1_layer_size = get_layer_neurals(-2);
    int hidden2_layer_size = get_layer_neurals(-3);
    int output_layer_size  = get_layer_neurals(-1);

    int _y_start_output = get_y_start(-1);
    int _x_start_output = get_x_start(-1);
    int _w_start_output = get_w_start(-1);
    int _b_start_hidden1 = get_b_start(-2);
    int _w_start_hidden1 = get_w_start(-2);
    int _x_start_hidden1 = get_x_start(-2);
    int _y_start_hidden2 = get_y_start(-3);

    //倒数第2隐含层，逐个更新神经元与下一层所有神经元的连接
    for (int hidden2_cell =0; hidden2_cell<hidden2_layer_size; ++hidden2_cell) {
        //逐个遍历下一层所有神经元
        for (int hidden1_cell=0; hidden1_cell<hidden1_layer_size; ++hidden1_cell) {
            //当前神经元与输出层链接，计算sum((y-Y)*f'(net)*w)
            double sum = 0.;
            for (int output_cell=0; output_cell<layers[n_layers-1]; ++output_cell) {
                sum += (vals[_y_start_output+output_cell]-Y[output_cell])
                        *active(vals[_x_start_output+output_cell])
                        *weights[_w_start_output+hidden1_cell*output_layer_size+output_cell];
            }
            //获得d_b
            d_bias[_b_start_hidden1+hidden1_cell] = sum*active_deri(vals[_x_start_hidden1+hidden1_cell]);
            d_weights[_w_start_hidden1+hidden2_cell*hidden1_layer_size+hidden1_cell]
                    = d_bias[_b_start_hidden1+hidden1_cell] * vals[_y_start_hidden2+hidden2_cell];
        }
    }
}

void BP::bf_hidden2_input() {
    //循环遍历每一层
    for (int cur_layer=n_layers-3; cur_layer>=0; --cur_layer) {
        int cur_layer_size = get_layer_neurals(cur_layer);
        int pre_layer_size = get_layer_neurals(cur_layer-1);
        int next_layer_size = get_layer_neurals(cur_layer+1);
        //上一层每个神经元
        int _b_start_cur = get_b_start(cur_layer);
        int _b_start_next = get_b_start(cur_layer+1);
        int _w_start_cur = get_w_start(cur_layer);
        int _w_start_next = get_w_start(cur_layer+1);
        int _x_start_cur = get_x_start(cur_layer);
        int _y_start_pre = get_y_start(cur_layer-1);

        for (int pre_hidden_cell=0; pre_hidden_cell<pre_layer_size; ++pre_hidden_cell) {
            //与当前层每个神经元连接的权重
            for (int cur_hidden_cell=0; cur_hidden_cell<cur_layer_size; ++cur_hidden_cell) {
                double sum = 0.;
                for (int next_hidden_cell=0; next_hidden_cell<next_layer_size; ++next_hidden_cell) {
                    sum += d_bias[_b_start_next+next_hidden_cell]
                            *weights[_w_start_next+cur_hidden_cell*next_layer_size+next_hidden_cell];
                }
                d_bias[_b_start_cur+cur_hidden_cell] = sum*active_deri(vals[_x_start_cur+cur_hidden_cell]);
                d_weights[_w_start_cur+pre_hidden_cell*cur_layer_size+cur_hidden_cell]
                        =d_bias[_b_start_cur+cur_hidden_cell]*vals[_y_start_pre+pre_hidden_cell];
            }
        }
    }
}

void BP::backword_flow() {
    bf_output_hidden1();
    bf_hidden1_hidden2();
    if (n_layers == 3) return;
    bf_hidden2_input();
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
    //assert(cur_layer>=0 && cur_layer <n_layers);
    return x_index[(cur_layer+n_layers)%n_layers];
}

int BP::get_y_start(int cur_layer) {
    //assert(cur_layer>=0 && cur_layer<n_layers);
    return y_index[(cur_layer+n_layers)%n_layers];
}

int BP::get_w_start(int cur_layer) {
    //assert(cur_layer>0 && cur_layer<n_layers);
    assert(cur_layer !=0);
    return w_index[(cur_layer+n_layers)%n_layers];
}

int BP::get_b_start(int cur_layer) {
    //assert(cur_layer>0 && cur_layer<n_layers);
    assert(cur_layer!=0);
    return b_index[(cur_layer+n_layers)%n_layers];
}

int BP::get_layer_neurals(int cur_layer) {
    return layers[(cur_layer+n_layers)%n_layers];
}

void BP::get_loss() {
    cur_loss = 0.;
    int _y_start_output = get_y_start(-1);
    int output_layer_size = get_layer_neurals(-1);
    for (int i=0; i<output_layer_size; ++i) {
        cur_loss += (Y[i]-vals[_y_start_output+i])*(Y[i]-vals[_y_start_output+i]);
    }
}

void BP::train(int _batch, int _max_itr_all,
               int _max_itr_batch, double _learning_rate, double _min_loss, string _GD,
               string _activation, bool _verbose) {
    batch = _batch;
    max_itr_all = _max_itr_all;
    max_itr_batch = _max_itr_batch;
    learning_rate = _learning_rate;
    min_loss = _min_loss;
    GD = _GD;
    activation = _activation;
    verbose = _verbose;


    for (int k=0; k<max_itr_all; ++k) {
        for (int i=0; i<(int)train_X.size(); ++i) {
            set_X(train_X[i]);
            set_Y(train_Y[i]);
            int __itr = 0;
            cur_loss = DBL_MAX_10_EXP;
            while (cur_loss > min_loss && __itr<max_itr_batch) {
                ++__itr;
                forword_flow();
                get_loss();
                if (verbose) {
                    cout <<"\tglobal_itr:"<<k+1<<"/"<<max_itr_all
                         <<"\tlocal_itr:"<<i+1<<"/"<<(int)train_X.size()<< "\tcur loss:" << cur_loss <<endl;
                }
                backword_flow();
                update_weights_bias();
            }
        }
    }
}

void BP::update_weights_bias() {
    for (int i=0; i<int(weights.size()); ++i) weights[i]-=learning_rate*d_weights[i];
    for (int i=0; i<int(bias.size()); ++i) bias[i]-=learning_rate*d_bias[i];
}

void BP::set_train_data(vector<vector<double>> X, vector<vector<double>> Y, double r) {
    assert(X.size()==Y.size());
    assert((int)X[0].size()==layers[0]);
    assert((int)Y[0].size()==layers.back());

    int num = X.size()*r;
    train_X = vector<vector<double>>(begin(X), begin(X)+num);
    train_Y = vector<vector<double>>(begin(Y), begin(Y)+num);
    test_X = vector<vector<double>>(begin(X)+num, end(X));
    test_Y = vector<vector<double>>(begin(Y)+num, end(Y));
}

void BP::predict(bool _verbose) {
    int correct_answer = 0;
    int _y_start_output = get_y_start(-1);
    int output_layer_size = get_layer_neurals(-1);

    for (int i=0; i<(int)test_X.size(); ++i) {
        set_X(test_X[i]);
        set_Y(test_Y[i]);
        forword_flow();
        //get_loss();
        int pred=0, real=0;
        double p=vals[_y_start_output], r=Y[0];
        for (int k=1; k<output_layer_size; ++k) {
            if (p<vals[_y_start_output+k]) {
                pred = k;
                p = vals[_y_start_output+k];
            }
            if (r<Y[k]) {
                real = k;
                r = Y[k];
            }
        }
        correct_answer += (pred==real);
        if (_verbose) cout <<GD<<"\t"<<pred<<"/"<<real<<"\tcur loss:"<<cur_loss<<"\taccuracy:"<<correct_answer/double(i+1)<<endl;
    }
    cout << "accuracy:"<<correct_answer/double(test_X.size()) <<endl;
}

void BP::data_normalization() {
    data_normal(test_X);
    data_normal(train_X);
}
