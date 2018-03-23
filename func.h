#ifndef FUNC_H
#define FUNC_H
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <assert.h>
using namespace std;

template<class T>
void print(vector<T> x) {
    cout << "[";
    for (auto i:x) cout << i << ", ";
    cout << "]" <<endl;
}

void read_mnist(vector<vector<double>>&, vector<vector<double>>&, string);

void data_normal(vector<vector<double>>&);
#endif // FUNC_H
