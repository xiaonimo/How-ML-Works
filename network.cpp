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
        b_index[i] = b_index[i-1]+layers[i-1];
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

    sigmoid_table = vector<double>(4000, 0.);
    init_sigmoid_table();
}

void BP::init_weights_bias() {
    //根据正太分布初始化参数
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
        const int _x_start_next = get_x_start(cur_layer+1);
        const int _y_start_cur = get_y_start(cur_layer);
        const int _w_start_next = get_w_start(cur_layer+1);
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
    const int hidden1_layer_size = get_layer_neurals(-2);
    const int output_layer_size = get_layer_neurals(-1);

    const int _w_start_output = get_w_start(-1);//最后一层（输出层）对应的权重连接
    const int _b_start_output = get_b_start(-1);
    const int _x_start_output = get_x_start(-1);
    const int _y_start_output = get_y_start(-1);
    const int _y_start_hidden1 = get_y_start(-2);
    //先计算d_bias
    for (int output_cell=0; output_cell<output_layer_size; ++output_cell) {
        d_bias[_b_start_output+output_cell] = (vals[_y_start_output+output_cell]-Y[output_cell])
                *active_deri(vals[_x_start_output+output_cell]);
    }

    //更新隐含层每个神经元与所有输出层神经元的权重
    for (int hidden1_cell=0; hidden1_cell<hidden1_layer_size; ++hidden1_cell) {
        //隐含层单个神经元与每个输出层神经元相连
        for (int output_cell=0; output_cell<output_layer_size; ++output_cell) {
            int __w_index = _w_start_output + hidden1_cell*output_layer_size + output_cell;
            d_weights[__w_index] = d_bias[_b_start_output+output_cell]*vals[_y_start_hidden1+hidden1_cell];
        }
    }
}

void BP::bf_hidden1_input() {
    //循环遍历每一层
    for (int cur_layer=n_layers-2; cur_layer>0; --cur_layer) {
        const int cur_layer_size = get_layer_neurals(cur_layer);
        const int pre_layer_size = get_layer_neurals(cur_layer-1);
        const int next_layer_size = get_layer_neurals(cur_layer+1);
        //上一层每个神经元
        const int _b_start_cur = get_b_start(cur_layer);
        const int _b_start_next = get_b_start(cur_layer+1);
        const int _w_start_cur = get_w_start(cur_layer);
        const int _w_start_next = get_w_start(cur_layer+1);
        const int _x_start_cur = get_x_start(cur_layer);
        const int _y_start_pre = get_y_start(cur_layer-1);
        //先计算d_bias
        for (int cur_hidden_cell=0; cur_hidden_cell<cur_layer_size; ++cur_hidden_cell) {
            double sum = 0.;
            for (int next_hidden_cell=0; next_hidden_cell<next_layer_size; ++next_hidden_cell) {
                sum += d_bias[_b_start_next+next_hidden_cell]
                        *weights[_w_start_next+cur_hidden_cell*next_layer_size+next_hidden_cell];
            }
            d_bias[_b_start_cur+cur_hidden_cell] = sum*active_deri(vals[_x_start_cur+cur_hidden_cell]);
        }

        //计算d_weights
        for (int pre_hidden_cell=0; pre_hidden_cell<pre_layer_size; ++pre_hidden_cell) {
            //与当前层每个神经元连接的权重
            for (int cur_hidden_cell=0; cur_hidden_cell<cur_layer_size; ++cur_hidden_cell) {
                d_weights[_w_start_cur+pre_hidden_cell*cur_layer_size+cur_hidden_cell]
                        =d_bias[_b_start_cur+cur_hidden_cell]*vals[_y_start_pre+pre_hidden_cell];
            }
        }
    }
}

void BP::backword_flow() {
    bf_output_hidden1();
    bf_hidden1_input();
}

double BP::active(double x) {
    if (activation == string("sigmoid")) {
        //return 1.0/(1+exp(-x));

        if (x>=20) return 0.999999999999;
        else if (x <= -20) return 0.0000000000001;
        else return sigmoid_table[int((x+20)*100)];

    }
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
    cur_loss/=2.0;
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

    mt19937_64 gen;
    uniform_int_distribution<int> gen_random_int(0, int(train_X.size())-1);

    if (GD == string("SGD")) {
    for (int k=0; k<max_itr_all; ++k) {
        for (int i=0; i<(int)train_X.size(); ++i) {
            int r_index = gen_random_int(gen);
            set_X(train_X[r_index]);
            set_Y(train_Y[r_index]);
            int __itr = 0;
            cur_loss = DBL_MAX_10_EXP;
            while (cur_loss > min_loss && __itr<max_itr_batch) {
                ++__itr;
                forword_flow();
                get_loss();
                if (verbose) {
                    cout <<"\tglobal_itr:"<<k+1<<"/"<<max_itr_all
                         <<"\tcur_items:"<<i+1<<"/"<<(int)train_X.size()<<"\tlocal_itr:"<<__itr
                         <<"\tcur loss:" << cur_loss <<endl;
                }
                backword_flow();
                update_weights_bias();
            }
        }
    }}
    else if (GD == string("BGD")) {
    //int _output_layer_size = get_layer_neurals(-1);
    int weights_num = int(weights.size());
    int bias_num = int(bias.size());
    for (int k=0; k<max_itr_all; ++k) {
        int batches = train_X.size()/batch;
        for (int i=0; i<batches; ++i) {
            int __itr = 0;
            double loss_batch = DBL_MAX_10_EXP;
            while (loss_batch > min_loss && __itr<max_itr_batch) {
            vector<double> d_bias_batch(bias_num, 0);
            vector<double> d_weights_batch(weights_num, 0);
            loss_batch = 0.;
            for (int j=0; j<batch; ++j) {
                int r_index = gen_random_int(gen);
                set_X(train_X[r_index]);
                set_Y(train_Y[r_index]);
                forword_flow();
                get_loss();
                loss_batch += cur_loss/batch;
                backword_flow();
                for (int _w=0; _w<weights_num; ++_w) d_weights_batch[_w]+=d_weights[_w]/batch;
                for (int _b=0; _b<bias_num; ++_b) d_bias_batch[_b]+=d_bias[_b]/batch;
            }
            for (int _w=0; _w<weights_num; ++_w) d_weights[_w] = d_weights_batch[_w];
            for (int _b=0; _b<bias_num; ++_b) d_bias[_b] = d_bias_batch[_b];
            update_weights_bias();
            ++__itr;
            if (verbose) {
                cout <<"\tglobal_itr:"<<k+1<<"/"<<max_itr_all
                     <<"\tcur_batch:"<<i+1<<"/"<<batches<<"\tlocal_itr:"<<__itr
                     <<"\tcur loss:" << loss_batch <<endl;
            }
            }
        }

    }}
    else {
        cout << "GD error" << endl;
        exit(0);
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
    const int _y_start_output = get_y_start(-1);
    const int output_layer_size = get_layer_neurals(-1);

    for (int i=0; i<(int)test_X.size(); ++i) {
        set_X(test_X[i]);
        set_Y(test_Y[i]);
        forword_flow();
        get_loss();
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


void BP::init_sigmoid_table() {
    double __s = 40.0/4000.0;
    for (int i=0; i<4000; ++i) {
        sigmoid_table[i] = 1.0/(1+exp(20-i*__s));
    }
    cout << "init sigmoid table finished" <<endl;
}

void BP::save_model(string filename) {
    ofstream fw(filename);
    if (!fw.is_open()) {
        cout << filename << " not opened!" <<endl;
        exit(0);
    }
    //保存网络结构
    for (int i=0; i<n_layers; ++i) {
        fw << layers[i];
        if (i!=n_layers-1) fw <<",";
        else fw << endl;
    }
    //保存激活函数
    fw << activation <<endl;
    //保存权重
    for (int i=0; i<(int)weights.size(); ++i) {
        fw << weights[i];
        if (i!=(int)weights.size()-1) fw<<",";
        else fw<<endl;
    }
    //保存bias
    for (int i=0; i<(int)bias.size(); ++i) {
        fw << bias[i];
        if (i!=(int)bias.size()-1) fw<<",";
        else fw<<endl;
    }
    fw.close();
}

void BP::load_model(string filename) {
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
    BP m(layer);

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






















