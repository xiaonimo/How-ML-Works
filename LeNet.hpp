#ifndef LENET_H
#define LENET_H
#include <array>
#include <ctime>
#include <cmath>
#include <random>
#include <cfloat>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#include "func.hpp"
#include "datatype.hpp"

class LeNet {
public:
    LeNet(std::vector<std::array<std::array<double, 32>, 32>> &_train_x, std::vector<std::array<double, 10>> &_train_y,
          std::vector<std::array<std::array<double, 32>, 32>> &_test_x,  std::vector<std::array<double, 10>> &_test_y,
          const unsigned _n_epoch=20, const unsigned _n_batch=1,
          const param_t _learning_rate=0.001, const param_t _min_loss=0.01, bool _verbose=true):
          train_x(_train_x), test_x(_test_x), train_y(_train_y), test_y(_test_y),
          n_epoch(_n_epoch), n_batch(_n_batch), n_step(train_x.size()/n_batch),
          learning_rate(_learning_rate), min_loss(_min_loss), verbose(_verbose){

        init_weights();
        init_sigmoid_table();
    }
    void fit();
    std::vector<index_t> predict(const std::vector<std::array<std::array<double, 32>, 32>> &x);

public:
    std::vector<std::array<std::array<double, 32>, 32>> &train_x, &test_x;
    std::vector<std::array<double, 10>> &train_y, &test_y;
    const unsigned n_epoch, n_batch, n_step;
    const param_t learning_rate, min_loss;
    bool verbose;

public:
    //layers
    std::array<std::array<data_t, 32>, 32> X;
    std::array<std::array<std::array<data_t, 28>, 28>, 6> C1, d_C1;
    std::array<std::array<std::array<data_t, 14>, 14>, 6> S2, d_S2;
    std::array<std::array<std::array<data_t, 10>, 10>, 16> C3, d_C3;
    std::array<std::array<std::array<data_t, 5>, 5>, 16> S4, d_S4;
    std::array<data_t, 120> C5, d_C5;
    std::array<data_t, 84> F6;
    std::array<data_t, 10> F7;
    std::array<data_t, 10> Y;

    //将C3<10*10> padding处理为C3_p<18*18>
    std::array<std::array<std::array<double, 18>, 18>, 16> C3_p;

    //kernels
    std::array<std::array<std::array<data_t, 5>, 5>, 6> k1;
    std::array<std::array<std::array<data_t, 5>, 5>, 16> k2, k2_r;     //k2_r是rotate 180的卷积核
    std::array<std::array<std::array<data_t, 5>, 5>, 120> k3;
    std::array<data_t, 6> kb1;
    std::array<data_t, 16> kb2;
    std::array<data_t, 120> kb3;

    //used in pooling(not uesd)
    std::array<data_t, 6> sw1;
    std::array<data_t, 6> sb1;
    std::array<data_t, 16> sw2;
    std::array<data_t, 16> sb2;

    //full-connected layer weights and bias
    std::array<std::array<data_t, 84>, 120> w1, d_w1;
    std::array<std::array<data_t, 10>, 84> w2, d_w2;
    std::array<data_t, 84> b1, d_b1;
    std::array<data_t, 10> b2, d_b2;

public:
    void init_weights();
    void forword_flow();
    void backword_flow();
    void update_weights();
    void X_to_C1();
    void C1_to_S2();
    void S2_to_C3();
    void C3_to_S4();
    void S4_to_C5();
    void C5_to_F6();
    void F6_to_F7();

    void F7_to_F6();
    void F6_to_C5();
    void C5_to_S4();
    void S4_to_C3();
    void C3_to_S2();
    void S2_to_C1();
    void C1_to_X();

    template<typename T, unsigned t1, unsigned t2>
    void print(const std::array<std::array<T, t1>, t2>&, bool verbose=true);
    template<typename T, unsigned t1>
    void print(const std::array<T, t1>&, bool verbose=true);
    void set_XY(const index_t i) {
        cur_index=i;
        X = train_x[cur_index];
        Y = train_y[cur_index];
    }

private:
    //used in forword_flow in S2_to_C3()
    data_t _conv3(unsigned, unsigned, unsigned, unsigned, int, int);
    data_t _conv4(unsigned, unsigned, unsigned, unsigned, unsigned, int, int);
    data_t _conv6(unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, int, int);

    //used in backword_flow in S2_to_C1()
    void _up(unsigned m, unsigned k);

    //used in backword_flow in update kernels
    void _d_kernels1(unsigned k); // 更新第k个kernel
    void _d_kernels2(unsigned k, unsigned x); // 更新第k个kernel, x是上一层的feature_map[x], 敏感项是下一层的d_X[k]

private:
    double cur_loss;
    index_t cur_index;
    std::array<data_t, 4000> sigmoid_table;

    void init_sigmoid_table();
    double active(double);
    double active_deri(double);
};

template<typename T, unsigned t1, unsigned t2>
void LeNet::print(const std::array<std::array<T, t1>, t2> &x, bool verbose) {
    if (verbose) {
        std::cout << std::endl << "shape:(" << t1 <<", " << t2 << ")" << std::endl;
    }
    std::cout << "[";
    for (index_t i=0; i<t2; ++i) {
        if (i) std::cout << " ";
        std::cout << "[";
        for (index_t j=0; j<t1; ++j) {
            std::cout << x[i][j];
            if (j!=t1-1) std::cout<<",";
        }
        std::cout << "]";
        if (i!=t2-1) std::cout<<std::endl;
    }
    std::cout << "]" << std::endl;
}

template<typename T, unsigned t1>
void LeNet::print(const std::array<T, t1> &x, bool verbose) {
    if (verbose) {
        std::cout << std::endl << "shape:(" << t1 <<", " << 1 << ")" << std::endl;
    }
    std::cout << "[";
    for (index_t i=0; i<t1; ++i) {
        std::cout << x[i];
        if (i != t1-1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
}

void
LeNet::fit() {
    for (index_t e=0; e<n_epoch; ++e) {
        nn::random_shuffle(train_x, train_y);
        for (index_t s=0; s<n_step; ++s) {
            auto t1 = clock();
            int _itr = 0;
            set_XY(s);
            cur_loss = min_loss+1;
            while (_itr++<500 && cur_loss > min_loss) {
                cur_loss = 0.;
                forword_flow();
                backword_flow();
                update_weights();
            }
            auto t2 = clock();
            if (verbose) {
                std::cout << "epoch:" << e+1 << " step:" << s+1 << " loss:" << cur_loss <<" time:" << t2-t1 << std::endl;
            }
        }
        auto res = predict(test_x);
        unsigned ca = 0;
        for(index_t i=0; i<test_x.size(); ++i) {
            ca += res[i] == nn::argmax(test_y[i]);
        }
        std::cout << "accuracy:" << double(ca)/test_x.size() << std::endl;
    }
}

std::vector<index_t>
LeNet::predict(const std::vector<std::array<std::array<double, 32>, 32>> &test_x) {
    std::vector<index_t> res;
    for (index_t i=0; i<test_x.size(); ++i) {
        X = test_x[i];
        forword_flow();
        print(F7);
        res.push_back(nn::argmax(F7));
    }
    return res;
}

void
LeNet::init_weights() {
    std::mt19937 gen;
    std::normal_distribution<double> normal(-0.01, 0.01);
    for(auto &_k:k1) for(auto &_r:_k) for(auto &_e:_r) _e=normal(gen);
    for(auto &_k:k2) for(auto &_r:_k) for(auto &_e:_r) _e=normal(gen);
    for(auto &_k:k3) for(auto &_r:_k) for(auto &_e:_r) _e=normal(gen);

    for(auto &_e:kb1) _e=normal(gen);
    for(auto &_e:kb2) _e=normal(gen);
    for(auto &_e:kb3) _e=normal(gen);

    for(auto &_w:w1) for(auto &_e:_w) _e=normal(gen);
    for(auto &_w:w2) for(auto &_e:_w) _e=normal(gen);

    for(auto &_e:b1) _e=normal(gen);
    for(auto &_e:b2) _e=normal(gen);
}

void
LeNet::backword_flow() {
    //auto t1 = clock();
    F7_to_F6();
    //auto t2 = clock();
    F6_to_C5();
    //auto t3 = clock();
    C5_to_S4();
    //auto t4 = clock();
    S4_to_C3();
    //auto t5 = clock();
    C3_to_S2();
    //auto t6 = clock();
    S2_to_C1();
    //auto t7 = clock();
    C1_to_X();
    //auto t8 = clock();

    //cout << t2-t1 << "\t"<< t3-t2 << "\t"<< t4-t3 << "\t"<< t5-t4 << "\t"<< t6-t5 << "\t"<< t7-t6 << "\t"<< t8-t7 << endl;
}

void
LeNet::F7_to_F6() {
    for (index_t i=0; i<10; ++i) cur_loss += std::pow(F7[i]-Y[i], 2);
    for (index_t i=0; i<10; ++i) d_b2[i] = F7[i]-Y[i];
    for (index_t i=0; i<84; ++i) for (index_t j=0; j<10; ++j) d_w2[i][j] = d_b2[j]*F6[i];
}

void
LeNet::F6_to_C5() {
    for (index_t i=0; i<84; ++i) {
        double sum=0.;
        for (index_t j=0; j<10; ++j) {
            sum += d_b2[j]*w2[i][j];
        }
        d_b1[i] = sum * active_deri(F6[i]);
    }
    for (index_t i=0; i<120; ++i) {
        for (index_t j=0; j<84; ++j) d_w1[i][j] = d_b1[j]*C5[i];
    }
}

void
LeNet::C5_to_S4() {
    //update d_C5 sensitives
    for (index_t i=0; i<120; ++i) {
        double sum=0.;
        for (index_t j=0; j<84; ++j) sum += d_b1[j]*w1[i][j];
        d_C5[i] = sum * active_deri(C5[i]);
    }
}

void
LeNet::S4_to_C3() {
    // update d_S4 sensitives
    d_S4 = {0};

    std::array<std::array<double, 5>, 5> tmp = {0};
    for (index_t k=0; k<120; ++k) {
        for (index_t i=0; i<5; ++i) {
            for (index_t j=0; j<5; ++j) {
                tmp[i][j] += k3[k][i][j]*d_C5[k];
            }
        }
    }

    for (index_t k=0; k<16; ++k) {
        for (index_t i=0; i<5; ++i) {
            for (index_t j=0; j<5; ++j) {
                d_S4[k][i][j] = tmp[i][j]*active_deri(S4[k][i][j]);
            }
        }
    }
}

void
LeNet::C3_to_S2() {
    //update d_C3 sensitives
    for (index_t k=0; k<16; ++k) {
        for (index_t i=0; i<5; ++i) {
            for (index_t j=0; j<5; ++j) {
                d_C3[k][i*2+0][j*2+0] *= d_S4[k][i][j];
                d_C3[k][i*2+0][j*2+1] *= d_S4[k][i][j];
                d_C3[k][i*2+1][j*2+0] *= d_S4[k][i][j];
                d_C3[k][i*2+1][j*2+1] *= d_S4[k][i][j];
            }
        }
    }
}

void
LeNet::S2_to_C1() {
    //padding feature map
    for (index_t k=0; k<16; ++k) {
        nn::padding(C3_p[k], d_C3[k]);
    }
    //rotate kernel
    for (index_t k=0; k<16; ++k) {
        for (index_t i=0; i<5; ++i) {
            for (index_t j=0; j<5; ++j) k2_r[k][i][j] = k2[k][4-i][4-j];
        }
    }

    //update d_S2 sensitives
    d_S2 = {0};
    _up(0, 0);_up(0, 4);_up(0, 5);_up(0, 6);_up(0, 9);_up(0, 10);_up(0, 11);_up(0, 12);_up(0, 14);_up(0, 15);
    _up(1, 0);_up(1, 1);_up(1, 5);_up(1, 6);_up(1, 7);_up(1, 10);_up(1, 11);_up(1, 12);_up(1, 13);_up(1, 15);
    _up(2, 0);_up(2, 1);_up(2, 2);_up(2, 6);_up(2, 7);_up(2,  8);_up(2, 11);_up(2, 13);_up(2, 14);_up(2, 15);
    _up(3, 1);_up(3, 2);_up(3, 3);_up(3, 6);_up(3, 7);_up(3,  8);_up(3,  9);_up(3, 12);_up(3, 14);_up(3, 15);
    _up(4, 2);_up(4, 3);_up(4, 4);_up(4, 7);_up(4, 8);_up(4,  9);_up(4, 10);_up(4, 12);_up(4, 13);_up(4, 15);
    _up(5, 3);_up(5, 4);_up(5, 5);_up(5, 8);_up(5, 9);_up(5, 10);_up(5, 11);_up(5, 13);_up(5, 14);_up(5, 15);
}

void
LeNet::C1_to_X() {
    //update d_C1 sensitives
    for (index_t k=0; k<6; ++k) {
        for (index_t i=0; i<14; ++i) {
            for (index_t j=0; j<14; ++j) {
                d_C1[k][i*2+0][j*2+0] *= d_S2[k][i][j];
                d_C1[k][i*2+0][j*2+1] *= d_S2[k][i][j];
                d_C1[k][i*2+1][j*2+0] *= d_S2[k][i][j];
                d_C1[k][i*2+1][j*2+1] *= d_S2[k][i][j];
            }
        }
    }
}

void
LeNet::update_weights() {
    //F7
    for (index_t i=0; i<10; ++i) b2[i] -= learning_rate*d_b2[i];
    for (index_t i=0; i<84; ++i) for (index_t j=0; j<10; ++j) w2[i][j] -= learning_rate*d_w2[i][j];

    //F6
    for (index_t i=0; i<84; ++i) b1[i] -= learning_rate*d_b1[i];
    for (index_t i=0; i<120; ++i) for (index_t j=0; j<84; ++j) w1[i][j] -= learning_rate*d_w1[i][j];

    //C5
    //kernel-bias
    for (index_t i=0; i<120; ++i) kb3[i] -= learning_rate*d_C5[i];
    //kernel-weights
    std::array<std::array<double, 5>, 5> S4_sum={0};
    for (index_t i=0; i<5; ++i) {
        for (index_t j=0; j<5; ++j) {
            for (index_t k=0; k<16; ++k) {
                S4_sum[i][j] += S4[k][i][j];
            }
        }
    }
    for (index_t k=0; k<120; ++k) {
        for (index_t i=0; i<5; ++i) {
            for (index_t j=0; j<5; ++j) {
                k3[k][i][j] -= learning_rate*S4_sum[i][j]*d_C5[k];
            }
        }
    }

    //C3
    //kernel-bias
    for (index_t k=0; k<16; ++k) {
        double sum = k2[k][0][0]+k2[k][0][1]+k2[k][0][2]+k2[k][0][3]+k2[k][0][4]+
                     k2[k][1][0]+k2[k][1][1]+k2[k][1][2]+k2[k][1][3]+k2[k][1][4]+
                     k2[k][2][0]+k2[k][2][1]+k2[k][2][2]+k2[k][2][3]+k2[k][2][4]+
                     k2[k][3][0]+k2[k][3][1]+k2[k][3][2]+k2[k][3][3]+k2[k][3][4]+
                     k2[k][4][0]+k2[k][4][1]+k2[k][4][2]+k2[k][4][3]+k2[k][4][4];
        kb2[k] -= learning_rate*sum;
    }
    //kernels-weights
    _d_kernels2(0, 0); _d_kernels2(0, 1) ;_d_kernels2(0, 2);
    _d_kernels2(1, 1); _d_kernels2(1, 2) ;_d_kernels2(1, 3);
    _d_kernels2(2, 2); _d_kernels2(2, 3) ;_d_kernels2(2, 4);
    _d_kernels2(3, 3); _d_kernels2(3, 4) ;_d_kernels2(3, 5);
    _d_kernels2(4, 0); _d_kernels2(4, 4) ;_d_kernels2(4, 5);
    _d_kernels2(5, 0); _d_kernels2(5, 1) ;_d_kernels2(5, 5);
    _d_kernels2(6, 0); _d_kernels2(6, 1) ;_d_kernels2(6, 2); _d_kernels2(6, 3);
    _d_kernels2(7, 1); _d_kernels2(7, 2) ;_d_kernels2(7, 3); _d_kernels2(7, 4);
    _d_kernels2(8, 2); _d_kernels2(8, 3) ;_d_kernels2(8, 4); _d_kernels2(8, 5);
    _d_kernels2(9, 0); _d_kernels2(8, 3) ;_d_kernels2(9, 4); _d_kernels2(9, 5);
    _d_kernels2(10, 0);_d_kernels2(10, 1);_d_kernels2(10, 4);_d_kernels2(10, 5);
    _d_kernels2(11, 0);_d_kernels2(11, 1);_d_kernels2(11, 2);_d_kernels2(11, 5);
    _d_kernels2(12, 0);_d_kernels2(12, 1);_d_kernels2(12, 3);_d_kernels2(12, 4);
    _d_kernels2(13, 1);_d_kernels2(13, 2);_d_kernels2(13, 4);_d_kernels2(13, 5);
    _d_kernels2(14, 0);_d_kernels2(14, 2);_d_kernels2(14, 3);_d_kernels2(14, 5);
    _d_kernels2(15, 0);_d_kernels2(15, 1);_d_kernels2(15, 2);_d_kernels2(15, 3);_d_kernels2(15, 4);_d_kernels2(15, 5);

    //C1
    //kernel-bias
    for (index_t k=0; k<6; ++k) {
        double sum = k2[k][0][0]+k2[k][0][1]+k2[k][0][2]+k2[k][0][3]+k2[k][0][4]+
                     k2[k][1][0]+k2[k][1][1]+k2[k][1][2]+k2[k][1][3]+k2[k][1][4]+
                     k2[k][2][0]+k2[k][2][1]+k2[k][2][2]+k2[k][2][3]+k2[k][2][4]+
                     k2[k][3][0]+k2[k][3][1]+k2[k][3][2]+k2[k][3][3]+k2[k][3][4]+
                     k2[k][4][0]+k2[k][4][1]+k2[k][4][2]+k2[k][4][3]+k2[k][4][4];
        kb1[k] -= learning_rate*sum;
    }
    for (index_t k=0; k<6; ++k) {
        _d_kernels1(k);
    }
}

void
LeNet::_d_kernels1(unsigned k) {
    for (index_t i=0; i<5; ++i) {
        for (index_t j=0; j<5; ++j) {
            double sum=0.;
            for (index_t m=0; m<28; ++m) {
                for (index_t n=0; n<28; ++n) {
                    sum += d_C1[k][i][j]*X[m+i][n+j];
                }
            }
            k1[k][i][j] -= learning_rate*sum;
        }
    }
}

void
LeNet::_d_kernels2(unsigned k, unsigned x) {
    for (index_t i=0; i<5; ++i) {
        for (index_t j=0; j<5; ++j) {
            double sum=0.;
            for (index_t m=0; m<10; ++m) {
                for (index_t n=0; n<10; ++n) {
                    sum += d_C3[k][i][j]*S2[x][m+i][n+j];
                }
            }
            k1[k][i][j] -= learning_rate*sum;
        }
    }
}

void
LeNet::_up(unsigned m, unsigned k) {
    for (index_t i=0; i<14; ++i) {
        for (index_t j=0; j<14; ++j) {
            d_S2[m][i][j] +=
           (k2_r[k][0][0]*C3_p[k][i+0][j+0] + k2_r[k][0][1]*C3_p[k][i+0][j+1] + k2_r[k][0][2]*C3_p[k][i+0][j+2] + k2_r[k][0][3]*C3_p[k][i+0][j+3] + k2_r[k][0][4]*C3_p[k][i+0][j+4]+
            k2_r[k][1][0]*C3_p[k][i+1][j+0] + k2_r[k][1][1]*C3_p[k][i+1][j+1] + k2_r[k][1][2]*C3_p[k][i+1][j+2] + k2_r[k][1][3]*C3_p[k][i+1][j+3] + k2_r[k][1][4]*C3_p[k][i+1][j+4]+
            k2_r[k][2][0]*C3_p[k][i+2][j+0] + k2_r[k][2][1]*C3_p[k][i+2][j+1] + k2_r[k][2][2]*C3_p[k][i+2][j+2] + k2_r[k][2][3]*C3_p[k][i+2][j+3] + k2_r[k][2][4]*C3_p[k][i+2][j+4]+
            k2_r[k][3][0]*C3_p[k][i+3][j+0] + k2_r[k][3][1]*C3_p[k][i+3][j+1] + k2_r[k][3][2]*C3_p[k][i+3][j+2] + k2_r[k][3][3]*C3_p[k][i+3][j+3] + k2_r[k][3][4]*C3_p[k][i+3][j+4]+
            k2_r[k][4][0]*C3_p[k][i+4][j+0] + k2_r[k][4][1]*C3_p[k][i+4][j+1] + k2_r[k][4][2]*C3_p[k][i+4][j+2] + k2_r[k][4][3]*C3_p[k][i+4][j+3] + k2_r[k][4][4]*C3_p[k][i+4][j+4])*
            active_deri(S2[m][i][j]);
        }
    }
}

void
LeNet::forword_flow() {
    //auto t1 = clock();
    X_to_C1();
    //for (int i=0; i<6; ++i) print(C1[i]);
    //auto t2 = clock();
    C1_to_S2();
    //for (int i=0; i<6; ++i) print(S2[i]);
    //auto t3 = clock();
    S2_to_C3();
    //for (int i=0; i<16; ++i) print(C3[i]);
    //auto t4 = clock();
    C3_to_S4();
    //for (int i=0; i<16; ++i) print(S4[i]);
    //auto t5 = clock();
    S4_to_C5();
    //for (int i=0; i<120; ++i) cout << C5[i] << "\t";
    //auto t6 = clock();
    C5_to_F6();
    //auto t7 = clock();
    F6_to_F7();
    //auto t8 = clock();

    //cout << (t2-t1) << "\t" << (t3-t2) << "\t" << (t4-t3) << "\t" <<
            //(t5-t4) << "\t" << (t6-t5) << "\t" << (t7-t6) << "\t" <<
            //(t8-t7) << "\t" << (t8-t1) <<endl;
}

void
LeNet::X_to_C1() {
    for (index_t k=0; k<6; ++k) {
        for (index_t i=0; i<28; ++i) {
            for (index_t j=0; j<28; ++j) {
                C1[k][i][j]=k1[k][0][0]*X[i+0][j+0] + k1[k][0][1]*X[i+0][j+1] + k1[k][0][2]*X[i+0][j+2] + k1[k][0][3]*X[i+0][j+3] + k1[k][0][4]*X[i+0][j+4]+
                            k1[k][1][0]*X[i+1][j+0] + k1[k][1][1]*X[i+1][j+1] + k1[k][1][2]*X[i+1][j+2] + k1[k][1][3]*X[i+1][j+3] + k1[k][1][4]*X[i+1][j+4]+
                            k1[k][2][0]*X[i+2][j+0] + k1[k][2][1]*X[i+2][j+1] + k1[k][2][2]*X[i+2][j+2] + k1[k][2][3]*X[i+2][j+3] + k1[k][2][4]*X[i+2][j+4]+
                            k1[k][3][0]*X[i+3][j+0] + k1[k][3][1]*X[i+3][j+1] + k1[k][3][2]*X[i+3][j+2] + k1[k][3][3]*X[i+3][j+3] + k1[k][3][4]*X[i+3][j+4]+
                            k1[k][4][0]*X[i+4][j+0] + k1[k][4][1]*X[i+4][j+1] + k1[k][4][2]*X[i+4][j+2] + k1[k][4][3]*X[i+4][j+3] + k1[k][4][4]*X[i+4][j+4]+
                            kb1[k];
                C1[k][i][j]=active(C1[k][i][j]);
            }
        }
    }
}

void
LeNet::C1_to_S2() {
    d_C1 = {0};
    for (index_t k=0; k<6; ++k) {
        for (index_t i=0; i<14; ++i) {
            for (index_t j=0; j<14; ++j) {
                //S2[k][i][j] = max({C1[k][i*2+0][j*2+0], C1[k][i*2+0][j*2+1], C1[k][i*2+1][j*2+0], C1[k][i*2+1][j*2+1]});
                double _tmp = C1[k][i*2+0][j*2+0];
                index_t _i = i*2+0, _j = j*2+0;
                if (C1[k][i*2+0][j*2+1]>_tmp) {
                    _tmp = C1[k][i*2+0][j*2+1];
                    _j = j*2+1;
                }
                if (C1[k][i*2+1][j*2+0]>_tmp) {
                    _tmp = C1[k][i*2+1][j*2+0];
                    _i = i*2+1; _j = j*2+0;
                }
                if (C1[k][i*2+1][j*2+1]>_tmp) {
                    _tmp = C1[k][i*2+1][j*2+1];
                    _i = i*2+1; _j = j*2+1;
                }
                S2[k][i][j] = _tmp;
                d_C1[k][_i][_j] = 1;
                //S2[k][i][j] = active((C1[k][i*2+0][j*2+0]+C1[k][i*2+0][j*2+1]+C1[k][i*2+1][j*2+0]+C1[k][i*2+1][j*2+1])*sw1[k]+sb1[k]);
            }
        }
    }
}

void
LeNet::S2_to_C3() {
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[0][i][j]=active(_conv3(0, 0, 1, 2, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[1][i][j]=active(_conv3(1, 1, 2, 3, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[2][i][j]=active(_conv3(2, 2, 3, 4, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[3][i][j]=active(_conv3(3, 3, 4, 5, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[4][i][j]=active(_conv3(4, 0, 4, 5, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[5][i][j]=active(_conv3(5, 0, 1, 5, i, j));
        }
    }

    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[6][i][j]=active(_conv4(6, 0, 1, 2, 3, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[7][i][j]=active(_conv4(7, 1, 2, 3, 4, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[8][i][j]=active(_conv4(8, 2, 3, 4, 5, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[9][i][j]=active(_conv4(9, 3, 4, 5, 0, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[10][i][j]=active(_conv4(10, 4, 5, 0, 1, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[11][i][j]=active(_conv4(11, 5, 0, 1, 2, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[12][i][j]=active(_conv4(12, 0, 1, 3, 4, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[13][i][j]=active(_conv4(13, 1, 2, 4, 5, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[14][i][j]=active(_conv4(14, 0, 2, 3, 5, i, j));
        }
    }
    for (index_t i=0; i<10; ++i) {
        for (index_t j=0; j<10; ++j) {
            C3[15][i][j]=active(_conv6(15, 0, 1, 2, 3, 4, 5, i, j));
        }
    }

}

void
LeNet::C3_to_S4() {
    d_C3 = {0};
    for (index_t k=0; k<16; ++k) {
        for (index_t i=0; i<5; ++i) {
            for (index_t j=0; j<5; ++j) {
                //S4[k][i][j] = max({C3[k][i*2+0][j*2+0], C3[k][i*2+0][j*2+1], C3[k][i*2+1][j*2+0], C3[k][i*2+1][j*2+1]});
                double _tmp=C3[k][i*2+0][j*2+0];
                index_t _i=i*2+0, _j=j*2+0;
                if (C3[k][i*2+0][j*2+1]>_tmp) {
                    _tmp = C3[k][i*2+0][j*2+1];
                    _j = j*2+1;
                }
                if (C3[k][i*2+1][j*2+0]>_tmp) {
                    _tmp = C3[k][i*2+1][j*2+0];
                    _i = i*2+1; _j = j*2+0;
                }
                if (C3[k][i*2+1][j*2+1]>_tmp) {
                    _tmp = C3[k][i*2+1][j*2+1];
                    _i = i*2+1; _j = j*2+1;
                }
                S4[k][i][j] = _tmp;
                d_C3[k][_i][_j] = 1;
                //S4[k][i][j] = active((C3[k][i*2+0][j*2+0]+C3[k][i*2+0][j*2+1]+C3[k][i*2+1][j*2+0]+C3[k][i*2+1][j*2+1])*sw2[k]+sb2[k]);
            }
        }
    }
}

void
LeNet::S4_to_C5() {
    for (index_t k=0; k<120; ++k) {
        data_t sum = 0.;
        for (index_t m=0; m<16; ++m) {
            sum += k3[k][0][0]*S4[m][0][0]+k3[k][0][1]*S4[m][0][1]+k3[k][0][2]*S4[m][0][2]+k3[k][0][3]*S4[m][0][3]+k3[k][0][4]*S4[m][0][4]+
                   k3[k][1][0]*S4[m][1][0]+k3[k][1][1]*S4[m][1][1]+k3[k][1][2]*S4[m][1][2]+k3[k][1][3]*S4[m][1][3]+k3[k][1][4]*S4[m][1][4]+
                   k3[k][2][0]*S4[m][2][0]+k3[k][2][1]*S4[m][2][1]+k3[k][2][2]*S4[m][2][2]+k3[k][2][3]*S4[m][2][3]+k3[k][2][4]*S4[m][2][4]+
                   k3[k][3][0]*S4[m][3][0]+k3[k][3][1]*S4[m][3][1]+k3[k][3][2]*S4[m][3][2]+k3[k][3][3]*S4[m][3][3]+k3[k][3][4]*S4[m][3][4]+
                   k3[k][4][0]*S4[m][4][0]+k3[k][4][1]*S4[m][4][1]+k3[k][4][2]*S4[m][4][2]+k3[k][4][3]*S4[m][4][3]+k3[k][4][4]*S4[m][4][4];
        }
        C5[k] = active(sum + kb3[k]);
    }
}

void
LeNet::C5_to_F6() {
    for (index_t i=0; i<84; ++i) {
        data_t sum = 0.;
        for (index_t j=0; j<120; ++j) {
            sum += C5[j]*w1[j][i];
        }
        F6[i] = active(sum + b1[i]);
    }
}

void
LeNet::F6_to_F7() {

    for (index_t i=0; i<10; ++i) {
        data_t sum = 0.;
        for (index_t j=0; j<84; ++j) {
            sum += F6[j]*w2[j][i];
        }
        F7[i] = active(sum + b2[i]);
    }
}

data_t
LeNet::_conv3(unsigned k, unsigned m1, unsigned m2, unsigned m3, int _i, int _j) {
    int i = _i, j = _j;
    return
    k2[k][0][0]*S2[m1][i+0][j+0] + k2[k][0][1]*S2[m1][i+0][j+1] + k2[k][0][2]*S2[m1][i+0][j+2] + k2[k][0][3]*S2[m1][i+0][j+3] + k2[k][0][4]*S2[m1][i+0][j+4]+
    k2[k][1][0]*S2[m1][i+1][j+0] + k2[k][1][1]*S2[m1][i+1][j+1] + k2[k][1][2]*S2[m1][i+1][j+2] + k2[k][1][3]*S2[m1][i+1][j+3] + k2[k][1][4]*S2[m1][i+1][j+4]+
    k2[k][2][0]*S2[m1][i+2][j+0] + k2[k][2][1]*S2[m1][i+2][j+1] + k2[k][2][2]*S2[m1][i+2][j+2] + k2[k][2][3]*S2[m1][i+2][j+3] + k2[k][2][4]*S2[m1][i+2][j+4]+
    k2[k][3][0]*S2[m1][i+3][j+0] + k2[k][3][1]*S2[m1][i+3][j+1] + k2[k][3][2]*S2[m1][i+3][j+2] + k2[k][3][3]*S2[m1][i+3][j+3] + k2[k][3][4]*S2[m1][i+3][j+4]+
    k2[k][4][0]*S2[m1][i+4][j+0] + k2[k][4][1]*S2[m1][i+4][j+1] + k2[k][4][2]*S2[m1][i+4][j+2] + k2[k][4][3]*S2[m1][i+4][j+3] + k2[k][4][4]*S2[m1][i+4][j+4]+

    k2[k][0][0]*S2[m2][i+0][j+0] + k2[k][0][1]*S2[m2][i+0][j+1] + k2[k][0][2]*S2[m2][i+0][j+2] + k2[k][0][3]*S2[m2][i+0][j+3] + k2[k][0][4]*S2[m2][i+0][j+4]+
    k2[k][1][0]*S2[m2][i+1][j+0] + k2[k][1][1]*S2[m2][i+1][j+1] + k2[k][1][2]*S2[m2][i+1][j+2] + k2[k][1][3]*S2[m2][i+1][j+3] + k2[k][1][4]*S2[m2][i+1][j+4]+
    k2[k][2][0]*S2[m2][i+2][j+0] + k2[k][2][1]*S2[m2][i+2][j+1] + k2[k][2][2]*S2[m2][i+2][j+2] + k2[k][2][3]*S2[m2][i+2][j+3] + k2[k][2][4]*S2[m2][i+2][j+4]+
    k2[k][3][0]*S2[m2][i+3][j+0] + k2[k][3][1]*S2[m2][i+3][j+1] + k2[k][3][2]*S2[m2][i+3][j+2] + k2[k][3][3]*S2[m2][i+3][j+3] + k2[k][3][4]*S2[m2][i+3][j+4]+
    k2[k][4][0]*S2[m2][i+4][j+0] + k2[k][4][1]*S2[m2][i+4][j+1] + k2[k][4][2]*S2[m2][i+4][j+2] + k2[k][4][3]*S2[m2][i+4][j+3] + k2[k][4][4]*S2[m2][i+4][j+4]+

    k2[k][0][0]*S2[m3][i+0][j+0] + k2[k][0][1]*S2[m3][i+0][j+1] + k2[k][0][2]*S2[m3][i+0][j+2] + k2[k][0][3]*S2[m3][i+0][j+3] + k2[k][0][4]*S2[m3][i+0][j+4]+
    k2[k][1][0]*S2[m3][i+1][j+0] + k2[k][1][1]*S2[m3][i+1][j+1] + k2[k][1][2]*S2[m3][i+1][j+2] + k2[k][1][3]*S2[m3][i+1][j+3] + k2[k][1][4]*S2[m3][i+1][j+4]+
    k2[k][2][0]*S2[m3][i+2][j+0] + k2[k][2][1]*S2[m3][i+2][j+1] + k2[k][2][2]*S2[m3][i+2][j+2] + k2[k][2][3]*S2[m3][i+2][j+3] + k2[k][2][4]*S2[m3][i+2][j+4]+
    k2[k][3][0]*S2[m3][i+3][j+0] + k2[k][3][1]*S2[m3][i+3][j+1] + k2[k][3][2]*S2[m3][i+3][j+2] + k2[k][3][3]*S2[m3][i+3][j+3] + k2[k][3][4]*S2[m3][i+3][j+4]+
    k2[k][4][0]*S2[m3][i+4][j+0] + k2[k][4][1]*S2[m3][i+4][j+1] + k2[k][4][2]*S2[m3][i+4][j+2] + k2[k][4][3]*S2[m3][i+4][j+3] + k2[k][4][4]*S2[m3][i+4][j+4]+

    kb2[k];
}

data_t
LeNet::_conv4(unsigned k, unsigned m1, unsigned m2, unsigned m3, unsigned m4, int _i, int _j) {
    int i = _i, j = _j;
    return
    k2[k][0][0]*S2[m1][i+0][j+0] + k2[k][0][1]*S2[m1][i+0][j+1] + k2[k][0][2]*S2[m1][i+0][j+2] + k2[k][0][3]*S2[m1][i+0][j+3] + k2[k][0][4]*S2[m1][i+0][j+4]+
    k2[k][1][0]*S2[m1][i+1][j+0] + k2[k][1][1]*S2[m1][i+1][j+1] + k2[k][1][2]*S2[m1][i+1][j+2] + k2[k][1][3]*S2[m1][i+1][j+3] + k2[k][1][4]*S2[m1][i+1][j+4]+
    k2[k][2][0]*S2[m1][i+2][j+0] + k2[k][2][1]*S2[m1][i+2][j+1] + k2[k][2][2]*S2[m1][i+2][j+2] + k2[k][2][3]*S2[m1][i+2][j+3] + k2[k][2][4]*S2[m1][i+2][j+4]+
    k2[k][3][0]*S2[m1][i+3][j+0] + k2[k][3][1]*S2[m1][i+3][j+1] + k2[k][3][2]*S2[m1][i+3][j+2] + k2[k][3][3]*S2[m1][i+3][j+3] + k2[k][3][4]*S2[m1][i+3][j+4]+
    k2[k][4][0]*S2[m1][i+4][j+0] + k2[k][4][1]*S2[m1][i+4][j+1] + k2[k][4][2]*S2[m1][i+4][j+2] + k2[k][4][3]*S2[m1][i+4][j+3] + k2[k][4][4]*S2[m1][i+4][j+4]+

    k2[k][0][0]*S2[m2][i+0][j+0] + k2[k][0][1]*S2[m2][i+0][j+1] + k2[k][0][2]*S2[m2][i+0][j+2] + k2[k][0][3]*S2[m2][i+0][j+3] + k2[k][0][4]*S2[m2][i+0][j+4]+
    k2[k][1][0]*S2[m2][i+1][j+0] + k2[k][1][1]*S2[m2][i+1][j+1] + k2[k][1][2]*S2[m2][i+1][j+2] + k2[k][1][3]*S2[m2][i+1][j+3] + k2[k][1][4]*S2[m2][i+1][j+4]+
    k2[k][2][0]*S2[m2][i+2][j+0] + k2[k][2][1]*S2[m2][i+2][j+1] + k2[k][2][2]*S2[m2][i+2][j+2] + k2[k][2][3]*S2[m2][i+2][j+3] + k2[k][2][4]*S2[m2][i+2][j+4]+
    k2[k][3][0]*S2[m2][i+3][j+0] + k2[k][3][1]*S2[m2][i+3][j+1] + k2[k][3][2]*S2[m2][i+3][j+2] + k2[k][3][3]*S2[m2][i+3][j+3] + k2[k][3][4]*S2[m2][i+3][j+4]+
    k2[k][4][0]*S2[m2][i+4][j+0] + k2[k][4][1]*S2[m2][i+4][j+1] + k2[k][4][2]*S2[m2][i+4][j+2] + k2[k][4][3]*S2[m2][i+4][j+3] + k2[k][4][4]*S2[m2][i+4][j+4]+

    k2[k][0][0]*S2[m3][i+0][j+0] + k2[k][0][1]*S2[m3][i+0][j+1] + k2[k][0][2]*S2[m3][i+0][j+2] + k2[k][0][3]*S2[m3][i+0][j+3] + k2[k][0][4]*S2[m3][i+0][j+4]+
    k2[k][1][0]*S2[m3][i+1][j+0] + k2[k][1][1]*S2[m3][i+1][j+1] + k2[k][1][2]*S2[m3][i+1][j+2] + k2[k][1][3]*S2[m3][i+1][j+3] + k2[k][1][4]*S2[m3][i+1][j+4]+
    k2[k][2][0]*S2[m3][i+2][j+0] + k2[k][2][1]*S2[m3][i+2][j+1] + k2[k][2][2]*S2[m3][i+2][j+2] + k2[k][2][3]*S2[m3][i+2][j+3] + k2[k][2][4]*S2[m3][i+2][j+4]+
    k2[k][3][0]*S2[m3][i+3][j+0] + k2[k][3][1]*S2[m3][i+3][j+1] + k2[k][3][2]*S2[m3][i+3][j+2] + k2[k][3][3]*S2[m3][i+3][j+3] + k2[k][3][4]*S2[m3][i+3][j+4]+
    k2[k][4][0]*S2[m3][i+4][j+0] + k2[k][4][1]*S2[m3][i+4][j+1] + k2[k][4][2]*S2[m3][i+4][j+2] + k2[k][4][3]*S2[m3][i+4][j+3] + k2[k][4][4]*S2[m3][i+4][j+4]+

    k2[k][0][0]*S2[m4][i+0][j+0] + k2[k][0][1]*S2[m4][i+0][j+1] + k2[k][0][2]*S2[m4][i+0][j+2] + k2[k][0][3]*S2[m4][i+0][j+3] + k2[k][0][4]*S2[m4][i+0][j+4]+
    k2[k][1][0]*S2[m4][i+1][j+0] + k2[k][1][1]*S2[m4][i+1][j+1] + k2[k][1][2]*S2[m4][i+1][j+2] + k2[k][1][3]*S2[m4][i+1][j+3] + k2[k][1][4]*S2[m4][i+1][j+4]+
    k2[k][2][0]*S2[m4][i+2][j+0] + k2[k][2][1]*S2[m4][i+2][j+1] + k2[k][2][2]*S2[m4][i+2][j+2] + k2[k][2][3]*S2[m4][i+2][j+3] + k2[k][2][4]*S2[m4][i+2][j+4]+
    k2[k][3][0]*S2[m4][i+3][j+0] + k2[k][3][1]*S2[m4][i+3][j+1] + k2[k][3][2]*S2[m4][i+3][j+2] + k2[k][3][3]*S2[m4][i+3][j+3] + k2[k][3][4]*S2[m4][i+3][j+4]+
    k2[k][4][0]*S2[m4][i+4][j+0] + k2[k][4][1]*S2[m4][i+4][j+1] + k2[k][4][2]*S2[m4][i+4][j+2] + k2[k][4][3]*S2[m4][i+4][j+3] + k2[k][4][4]*S2[m4][i+4][j+4]+

    kb2[k];
}

data_t
LeNet::_conv6(unsigned k, unsigned m1, unsigned m2, unsigned m3, unsigned m4, unsigned m5, unsigned m6, int _i, int _j) {
    int i = _i, j = _j;
    return
    k2[k][0][0]*S2[m1][i+0][j+0] + k2[k][0][1]*S2[m1][i+0][j+1] + k2[k][0][2]*S2[m1][i+0][j+2] + k2[k][0][3]*S2[m1][i+0][j+3] + k2[k][0][4]*S2[m1][i+0][j+4]+
    k2[k][1][0]*S2[m1][i+1][j+0] + k2[k][1][1]*S2[m1][i+1][j+1] + k2[k][1][2]*S2[m1][i+1][j+2] + k2[k][1][3]*S2[m1][i+1][j+3] + k2[k][1][4]*S2[m1][i+1][j+4]+
    k2[k][2][0]*S2[m1][i+2][j+0] + k2[k][2][1]*S2[m1][i+2][j+1] + k2[k][2][2]*S2[m1][i+2][j+2] + k2[k][2][3]*S2[m1][i+2][j+3] + k2[k][2][4]*S2[m1][i+2][j+4]+
    k2[k][3][0]*S2[m1][i+3][j+0] + k2[k][3][1]*S2[m1][i+3][j+1] + k2[k][3][2]*S2[m1][i+3][j+2] + k2[k][3][3]*S2[m1][i+3][j+3] + k2[k][3][4]*S2[m1][i+3][j+4]+
    k2[k][4][0]*S2[m1][i+4][j+0] + k2[k][4][1]*S2[m1][i+4][j+1] + k2[k][4][2]*S2[m1][i+4][j+2] + k2[k][4][3]*S2[m1][i+4][j+3] + k2[k][4][4]*S2[m1][i+4][j+4]+

    k2[k][0][0]*S2[m2][i+0][j+0] + k2[k][0][1]*S2[m2][i+0][j+1] + k2[k][0][2]*S2[m2][i+0][j+2] + k2[k][0][3]*S2[m2][i+0][j+3] + k2[k][0][4]*S2[m2][i+0][j+4]+
    k2[k][1][0]*S2[m2][i+1][j+0] + k2[k][1][1]*S2[m2][i+1][j+1] + k2[k][1][2]*S2[m2][i+1][j+2] + k2[k][1][3]*S2[m2][i+1][j+3] + k2[k][1][4]*S2[m2][i+1][j+4]+
    k2[k][2][0]*S2[m2][i+2][j+0] + k2[k][2][1]*S2[m2][i+2][j+1] + k2[k][2][2]*S2[m2][i+2][j+2] + k2[k][2][3]*S2[m2][i+2][j+3] + k2[k][2][4]*S2[m2][i+2][j+4]+
    k2[k][3][0]*S2[m2][i+3][j+0] + k2[k][3][1]*S2[m2][i+3][j+1] + k2[k][3][2]*S2[m2][i+3][j+2] + k2[k][3][3]*S2[m2][i+3][j+3] + k2[k][3][4]*S2[m2][i+3][j+4]+
    k2[k][4][0]*S2[m2][i+4][j+0] + k2[k][4][1]*S2[m2][i+4][j+1] + k2[k][4][2]*S2[m2][i+4][j+2] + k2[k][4][3]*S2[m2][i+4][j+3] + k2[k][4][4]*S2[m2][i+4][j+4]+

    k2[k][0][0]*S2[m3][i+0][j+0] + k2[k][0][1]*S2[m3][i+0][j+1] + k2[k][0][2]*S2[m3][i+0][j+2] + k2[k][0][3]*S2[m3][i+0][j+3] + k2[k][0][4]*S2[m3][i+0][j+4]+
    k2[k][1][0]*S2[m3][i+1][j+0] + k2[k][1][1]*S2[m3][i+1][j+1] + k2[k][1][2]*S2[m3][i+1][j+2] + k2[k][1][3]*S2[m3][i+1][j+3] + k2[k][1][4]*S2[m3][i+1][j+4]+
    k2[k][2][0]*S2[m3][i+2][j+0] + k2[k][2][1]*S2[m3][i+2][j+1] + k2[k][2][2]*S2[m3][i+2][j+2] + k2[k][2][3]*S2[m3][i+2][j+3] + k2[k][2][4]*S2[m3][i+2][j+4]+
    k2[k][3][0]*S2[m3][i+3][j+0] + k2[k][3][1]*S2[m3][i+3][j+1] + k2[k][3][2]*S2[m3][i+3][j+2] + k2[k][3][3]*S2[m3][i+3][j+3] + k2[k][3][4]*S2[m3][i+3][j+4]+
    k2[k][4][0]*S2[m3][i+4][j+0] + k2[k][4][1]*S2[m3][i+4][j+1] + k2[k][4][2]*S2[m3][i+4][j+2] + k2[k][4][3]*S2[m3][i+4][j+3] + k2[k][4][4]*S2[m3][i+4][j+4]+

    k2[k][0][0]*S2[m4][i+0][j+0] + k2[k][0][1]*S2[m4][i+0][j+1] + k2[k][0][2]*S2[m4][i+0][j+2] + k2[k][0][3]*S2[m4][i+0][j+3] + k2[k][0][4]*S2[m4][i+0][j+4]+
    k2[k][1][0]*S2[m4][i+1][j+0] + k2[k][1][1]*S2[m4][i+1][j+1] + k2[k][1][2]*S2[m4][i+1][j+2] + k2[k][1][3]*S2[m4][i+1][j+3] + k2[k][1][4]*S2[m4][i+1][j+4]+
    k2[k][2][0]*S2[m4][i+2][j+0] + k2[k][2][1]*S2[m4][i+2][j+1] + k2[k][2][2]*S2[m4][i+2][j+2] + k2[k][2][3]*S2[m4][i+2][j+3] + k2[k][2][4]*S2[m4][i+2][j+4]+
    k2[k][3][0]*S2[m4][i+3][j+0] + k2[k][3][1]*S2[m4][i+3][j+1] + k2[k][3][2]*S2[m4][i+3][j+2] + k2[k][3][3]*S2[m4][i+3][j+3] + k2[k][3][4]*S2[m4][i+3][j+4]+
    k2[k][4][0]*S2[m4][i+4][j+0] + k2[k][4][1]*S2[m4][i+4][j+1] + k2[k][4][2]*S2[m4][i+4][j+2] + k2[k][4][3]*S2[m4][i+4][j+3] + k2[k][4][4]*S2[m4][i+4][j+4]+

    k2[k][0][0]*S2[m5][i+0][j+0] + k2[k][0][1]*S2[m5][i+0][j+1] + k2[k][0][2]*S2[m5][i+0][j+2] + k2[k][0][3]*S2[m5][i+0][j+3] + k2[k][0][4]*S2[m5][i+0][j+4]+
    k2[k][1][0]*S2[m5][i+1][j+0] + k2[k][1][1]*S2[m5][i+1][j+1] + k2[k][1][2]*S2[m5][i+1][j+2] + k2[k][1][3]*S2[m5][i+1][j+3] + k2[k][1][4]*S2[m5][i+1][j+4]+
    k2[k][2][0]*S2[m5][i+2][j+0] + k2[k][2][1]*S2[m5][i+2][j+1] + k2[k][2][2]*S2[m5][i+2][j+2] + k2[k][2][3]*S2[m5][i+2][j+3] + k2[k][2][4]*S2[m5][i+2][j+4]+
    k2[k][3][0]*S2[m5][i+3][j+0] + k2[k][3][1]*S2[m5][i+3][j+1] + k2[k][3][2]*S2[m5][i+3][j+2] + k2[k][3][3]*S2[m5][i+3][j+3] + k2[k][3][4]*S2[m5][i+3][j+4]+
    k2[k][4][0]*S2[m5][i+4][j+0] + k2[k][4][1]*S2[m5][i+4][j+1] + k2[k][4][2]*S2[m5][i+4][j+2] + k2[k][4][3]*S2[m5][i+4][j+3] + k2[k][4][4]*S2[m5][i+4][j+4]+

    k2[k][0][0]*S2[m6][i+0][j+0] + k2[k][0][1]*S2[m6][i+0][j+1] + k2[k][0][2]*S2[m6][i+0][j+2] + k2[k][0][3]*S2[m6][i+0][j+3] + k2[k][0][4]*S2[m6][i+0][j+4]+
    k2[k][1][0]*S2[m6][i+1][j+0] + k2[k][1][1]*S2[m6][i+1][j+1] + k2[k][1][2]*S2[m6][i+1][j+2] + k2[k][1][3]*S2[m6][i+1][j+3] + k2[k][1][4]*S2[m6][i+1][j+4]+
    k2[k][2][0]*S2[m6][i+2][j+0] + k2[k][2][1]*S2[m6][i+2][j+1] + k2[k][2][2]*S2[m6][i+2][j+2] + k2[k][2][3]*S2[m6][i+2][j+3] + k2[k][2][4]*S2[m6][i+2][j+4]+
    k2[k][3][0]*S2[m6][i+3][j+0] + k2[k][3][1]*S2[m6][i+3][j+1] + k2[k][3][2]*S2[m6][i+3][j+2] + k2[k][3][3]*S2[m6][i+3][j+3] + k2[k][3][4]*S2[m6][i+3][j+4]+
    k2[k][4][0]*S2[m6][i+4][j+0] + k2[k][4][1]*S2[m6][i+4][j+1] + k2[k][4][2]*S2[m6][i+4][j+2] + k2[k][4][3]*S2[m6][i+4][j+3] + k2[k][4][4]*S2[m6][i+4][j+4]+

    kb2[k];
}

void
LeNet::init_sigmoid_table() {
    double __s = 40.0/4000.0;
    for (index_t i=0; i<4000; ++i) {
        sigmoid_table[i] = 1.0/(1+exp(20-i*__s));
        //sigmoid_table[i] = tanh(i*__s-20);
    }
    std::cout << "init sigmoid table finished!" << std::endl;
}

double
LeNet::active(double x) {
    /*if (x>=10) return 1/(1+std::exp(-10));
    if (x<=-10) return 1/(1+std::exp(10));
    return 1/(1+std::exp(-x));*/

    if (x>=20) return 0.999999999999;
    else if (x <= -20) return 0.0000000000001;
    else return sigmoid_table[int((x+20)*100)];
    //else return 1/(1+exp(-x));
}

double
LeNet::active_deri(double y) {
    //x = active(x);
    //return x*(1-x);
    return y*(1-y);
    //return 1-pow(y, 2);
}

#endif // LENET_H
