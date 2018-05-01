
#include "rbf.hpp"
#include "mnist.hpp"
#include "kmeans.hpp"
#include <cstdio>

std::size_t argmax(const std::vector<double> &p) {
    std::size_t res = 0;
    double _max_v = p[res];

    for (index_t i=1; i<p.size(); ++i) {
        if (p[i] < _max_v) continue;
        _max_v = p[i];
        res = i;
    }
    return res;
}

void gen_data() {
    if (nullptr == std::freopen("data_mul.csv", "w", stdout)) {
        throw std::invalid_argument("open file failed!");
    }
    std::mt19937 gen;
    std::uniform_real_distribution<double> urd(-10, 10);
    /*
    for (int i=0; i<100; ++i) {
        for (int j=0; j<100; ++j) {
            for (int k=0; k<10; ++k) {
                double a = urd(gen);
                double b = urd(gen);
                double c = urd(gen);
                printf("%lf,%lf,%lf,%lf\n", a, b, c, a+b+c);
            }
        }
    }*/
    for (int i=0; i<20000; ++i) {
        double a = urd(gen);
        printf("%lf,%lf\n", a, a*a + std::cos(a));
    }
    std::fclose(stdout);
    //std::freopen("CON", "o", stdout);
    std::cout << "generate data finished!" << std::endl;
}

void read_data(std::vector<std::vector<data_t>>& X, std::vector<std::vector<data_t>>& Y, std::string filename) {
    if (X.size() != Y.size()) throw std::invalid_argument(" X and Y's size shoule be same!");
    unsigned num = X.size();
    unsigned dim = X[0].size();

    if(nullptr == std::freopen(filename.c_str(), "r", stdin)) {
        std::cout << "open " << filename << " failed!" <<std::endl;
    }
    data_t val = 0.;
    for (unsigned i=0; i<num; ++i) {
        for (unsigned j=0; j<dim+1; ++j) {
            scanf("%lf,", &val);
            if (j == dim) Y[i][0]=val;
            else X[i][j] = val;
        }
    }
    std::fclose(stdin);
    std::freopen("CON", "r", stdin);
    std::cout << "read " << filename << " finished!" << std::endl;
}

int main() {

    /* Classification
    const int num = 10000;
    points_t X(num, point_t(784));
    points_t Y(num, point_t(10));
    read_mnist(X, Y, "train.csv");
    data_normalization1(X);

    auto train_x = points_t(std::begin(X), std::begin(X)+int(num*0.85));
    auto train_y = points_t(std::begin(Y), std::begin(Y)+int(num*0.85));

    auto test_x = points_t(std::begin(X)+int(num*0.85), std::end(X));
    auto test_y = points_t(std::begin(Y)+int(num*0.85), std::end(Y));

    //Kmeans k(train_x, 1000);
    //k.cluster();

    rbf r(train_x, train_y, 784, 800, 10);
    r.train();
    auto pre = r.predict(test_x);

    int ca = 0;
    for (index_t i=0; i<test_y.size(); ++i) {
        ca += pre[i]==argmax(test_y[i]);
        std::cout << pre[i] << "/" << argmax(test_y[i]) <<std::endl;
    }
    std::cout << "accuracy:" << double(ca)/test_y.size() <<std::endl;
    */

    //Regression
    const int num = 2000;
    points_t X(num, point_t(1));
    points_t Y(num, point_t(1));
    read_data(X, Y, "data_mul.csv");
    data_normalization2(X);
    data_normalization2(Y);

    auto train_x = points_t(std::begin(X), std::begin(X)+int(num*0.85));
    auto train_y = points_t(std::begin(Y), std::begin(Y)+int(num*0.85));

    auto test_x = points_t(std::begin(X)+int(num*0.85), std::end(X));
    auto test_y = points_t(std::begin(Y)+int(num*0.85), std::end(Y));

    rbf r(train_x, train_y, 1, 10, 1);
    r.train();
    auto pre = r.predict_regression(test_x);
    for (index_t i=0; i<test_x.size(); ++i) {
        std::cout << pre[i][0] << "/" << test_y[i][0] <<std::endl;
    }

    return 0;
}


/*
#include<iostream>
#include<cassert>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<vector>
#include<iomanip>

using namespace std;

const int P=1000;        //输入样本的数量
vector<double> X(P);  //输入样本
vector<double> Y(P);      //输入样本对应的期望输出
const int M=10;         //隐藏层节点数目
vector<double> center(M);       //M个Green函数的数据中心
vector<double> delta(M);        //M个Green函数的扩展常数
double Green[P][M];         //Green矩阵
vector<double> Weight(M);       //权值矩阵
const double eta=0.001;     //学习率
const double ERR=0.9;       //目标误差
const int ITERATION_CEIL=5000;      //最大训练次数
vector<double> error(P);  //单个样本引起的误差

/*Hermit多项式函数*/
/*
inline double Hermit(double x){
    //return 1.1*(1-x+2*x*x)*exp(-1*x*x/2);
    return 2*x;
}

/*产生指定区间上均匀分布的随机数
inline double uniform(double floor,double ceil){
    return floor+1.0*rand()/RAND_MAX*(ceil-floor);
}

/*产生区间[floor,ceil]上服从正态分布N[mu,sigma]的随机数
inline double RandomNorm(double mu,double sigma,double floor,double ceil){
    double x,prob,y;
    do{
        x=uniform(floor,ceil);
        prob=1/sqrt(2*M_PI*sigma)*exp(-1*(x-mu)*(x-mu)/(2*sigma*sigma));
        y=1.0*rand()/RAND_MAX;
    }while(y>prob);
    return x;
}

/*产生输入样本
void generateSample(){
    for(int i=0;i<P;++i){
        double in=uniform(-4,4);
        X[i]=in;
        Y[i]=Hermit(in)+RandomNorm(0,0.1,-0.3,0.3);
        //Y[i] = in*in + 4;
    }
}

/*给向量赋予[floor,ceil]上的随机值
void initVector(vector<double> &vec,double floor,double ceil){
    for(int i=0;i<vec.size();++i)
        vec[i]=uniform(floor,ceil);
}

/*根据网络，由输入得到输出
double getOutput(double x){
    double y=0.0;
    for(int i=0;i<M;++i)
        y+=Weight[i]*exp(-1.0*(x-center[i])*(x-center[i])/(2*delta[i]*delta[i]));
    return y;
}

/*计算单个样本引起的误差
double calSingleError(int index){
    double output=getOutput(X[index]);
    return Y[index]-output;
}

/*计算所有训练样本引起的总误差
double calTotalError(){
    double rect=0.0;
    for(int i=0;i<P;++i){
        error[i]=calSingleError(i);
        rect+=error[i]*error[i];
    }
    return rect/2;
}

/*更新网络参数
void updateParam(){
    for(int j=0;j<M;++j){
        double delta_center=0.0,delta_delta=0.0,delta_weight=0.0;
        double sum1=0.0,sum2=0.0,sum3=0.0;
        for(int i=0;i<P;++i){
            sum1+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j]);
            sum2+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j])*(X[i]-center[j]);
            sum3+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]));
        }
        delta_center=eta*Weight[j]/(delta[j]*delta[j])*sum1;
        delta_delta=eta*Weight[j]/pow(delta[j],3)*sum2;
        delta_weight=eta*sum3;
        center[j]+=delta_center;
        delta[j]+=delta_delta;
        Weight[j]+=delta_weight;
    }
}

int main(int argc,char *argv[]){
    srand(time(0));
    /*初始化网络参数
    initVector(Weight,-0.1,0.1);
    initVector(center,-4.0,4.0);
    initVector(delta,0.1,0.3);
    /*产生输入样本
    generateSample();
    /*开始迭代
    int iteration=ITERATION_CEIL;
    while(iteration-->0){
        if(calTotalError()<ERR)      //误差已达到要求，可以退出迭代
            break;
        updateParam();      //更新网络参数
    }
    cout<<"迭代次数:"<<ITERATION_CEIL-iteration-1<<endl;

    //根据已训练好的神经网络作几组测试
    for(int x=-4;x<5;++x){
        cout<<x<<"\t";
        cout<<setprecision(8)<<setiosflags(ios::left)<<setw(15);
        cout<<getOutput(x)<<Hermit(x)<<endl;      //先输出我们预测的值，再输出真实值
    }
    return 0;
}
*/
