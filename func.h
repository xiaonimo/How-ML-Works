#ifndef FUNC_H
#define FUNC_H
#include <vector>
#include <iostream>
using namespace std;

template<class T>
void print(vector<T> x) {
    cout << "[";
    for (auto i:x) cout << i << ", ";
    cout << "]" <<endl;
}

#endif // FUNC_H
