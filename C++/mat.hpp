#ifndef MAT_H
#define MAT_H
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <algorithm> //accumulate
#include <stdexcept>

//tmd, 模板是真的不好写，算了，我还是不作死了，统统double算了
//而且除了输入层的像素外，其他都是浮点型
//doube能表示int，尽管有小小误差，就这样吧


class Mat {
    typedef double T;
    typedef std::vector<std::vector<T>> mat_t;
    typedef std::vector<T> vec_t;
    typedef std::size_t index_t;

public:
    Mat():rows(0), cols(0), data(mat_t()) {}
    Mat(vec_t _m):rows(1), cols(_m.size()), data(mat_t(1, _m)) {}
    Mat(mat_t _m):rows(_m.size()), cols(_m[0].size()), data(_m) {}
    Mat(int _rows, int _cols, T _val=T(0)):rows(_rows), cols(_cols), data(mat_t(_rows, vec_t(_cols, _val))) {}

    void print();                               //打印矩阵

    Mat operator *(const Mat&) const;                 //矩阵行列式计算
    Mat operator *(const T) const;                    //数乘

    Mat operator +(const Mat&) const;                 //矩阵各对应元素相加（矩阵行列数相同）
    Mat operator +(const T) const;                    //矩阵各元素增加一个常量

    Mat operator -(const Mat&) const;                 //矩阵各对应元素相减（矩阵行列数相同）
    Mat operator -(const T) const;                    //矩阵各元素减去一个常量

    Mat operator /(const T) const;                    //矩阵各元素除以一个常量

    vec_t& operator [](const index_t);                //用Mat[x][y]方式获取一个元素的值

    Mat mul(const Mat&) const;                        //矩阵对应元素相乘，返回一个新矩阵
    Mat square();                               //矩阵各元素取平方
    Mat inverse();                              //矩阵转置
    T sum();                                    //矩阵所有元素求和
    bool isEmpty() const {return rows==0 || cols==0;}

public:
    const unsigned int rows;                             //矩阵行数
    const unsigned int cols;                             //矩阵列数
    mat_t data;                                 //用二维vector数组存储矩阵元素
};

void Mat::print() {
    std::cout << "[";
    for (index_t r=0; r<rows; ++r) {
        if (r>0) std::cout << " ";//从第二行开始对齐
        if (cols>0) std::cout << "[";
        for (index_t c=0; c<cols; ++c) {
            std::cout << std::setprecision(10) << data[r][c];
            if (c < cols-1) std::cout << ",";
        }
        if (cols>0) std::cout << "]";
        if (r != rows-1) std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

Mat Mat::operator *(const Mat &b) const {
    if (cols != b.rows) throw std::invalid_argument("rows == cols!");
    if ((*this).isEmpty() || b.isEmpty()) throw std::invalid_argument("mat is empty");

    Mat res(rows, b.cols);
    for (index_t r=0; r<rows; ++r) {
        for (index_t c=0; c<b.cols; ++c) {
            T sum = T(0);
            for (index_t _e=0; _e<cols; ++_e) sum+=data[r][_e]*b.data[_e][c];
            res[r][c] = sum;
        }
    }
    return res;
}

Mat Mat::operator *(const T x) const{
    Mat res(rows, cols);
    for (index_t r=0; r<rows; ++r) {
        for (index_t c=0; c<cols; ++c) {
            res.data[r][c] = data[r][c]*x;
        }
    }
    return res;
}

Mat Mat::operator +(const Mat &b) const {
    if (rows != b.rows || cols != b.cols) throw std::invalid_argument("rows or cols are not equal");

    Mat res(rows, cols);
    for (index_t r=0; r<rows; ++r) {
        for (index_t c=0; c<cols; ++c) {
            res.data[r][c] = data[r][c]+b.data[r][c];
        }
    }
    return res;
}

Mat Mat::operator +(const T x) const {
    Mat res(rows, cols);
    for (index_t r=0; r<rows; ++r) {
        for (index_t c=0; c<cols; ++c) {
            res.data[r][c] = data[r][c]+x;
        }
    }
    return res;
}

Mat Mat::operator -(const Mat &b) const {
    return (*this) + (b*(-1));
}

Mat Mat::operator -(double x) const {
    return (*this) + (-x);
}


Mat Mat::operator /(double x) const {
    return (*this) * (1/x);
}

typename Mat::vec_t& Mat::operator [](const index_t x) {
    if (x>=rows) throw std::invalid_argument("row out of range");
    return data[x];
}

Mat Mat::inverse() {
    Mat res(cols, rows);
    for (index_t r=0;r<rows;++r) {
        for (index_t c=0;c<cols;++c) {
            res.data[c][r] = data[r][c];
        }
    }
    return res;
}

Mat Mat::mul(const Mat& b) const {
    if (cols != b.rows) throw std::invalid_argument("rows == cols!");
    if (this->isEmpty() || b.isEmpty()) throw std::invalid_argument("mat is empty");

    Mat res(rows, cols);
    for (index_t r=0; r<rows; ++r) {
        for (index_t c=0; c<cols; ++c) {
            res.data[r][c] = data[r][c]*b.data[r][c];
        }
    }
    return res;
}

Mat Mat::square() {
    Mat res(rows, cols);
    for (index_t r=0; r<rows; ++r) {
        for (index_t c=0; c<cols; ++c) {
            res.data[r][c] = data[r][c]*data[r][c];
        }
    }
    return res;
}

double Mat::sum() {
    T sum = T(0);
    for (index_t r=0; r<rows; ++r) {
        sum += accumulate(std::begin(data[r]), std::end(data[r]), T(0));
    }
    return sum;
}

#endif
