#ifndef RBF_H
#define RBF_H

#include <array>
using namespace std;

class rbf {
public:
    rbf(unsigned _n_input, unsigned _n_hidden, unsigned _n_output):
       n_input(_n_input), n_hidden(_n_hidden), n_output(_n_output){}
    void train();
    void predict();
private:
    unsigned n_input, n_hidden, n_output;
    array<double, n_hidden> center;
    array<double, n_hidden*n_output> weights;
};

#endif // RBF_H
