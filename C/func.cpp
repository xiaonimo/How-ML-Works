#include "func.h"

void read_mnist(vector<vector<double>>& X, vector<vector<double>>& Y, string filename) {
    assert(X.size()==Y.size());
    int num = X.size();

    freopen(filename.c_str(), "r", stdin);
    double val = 0.;
    for (int i=0; i<num; ++i) {
        for (int j=0; j<784+1; ++j) {
            scanf("%lf,", &val);
            if (j == 0) Y[i][int(val)] = 1;
            else X[i][j-1] = val;
        }
    }
    fclose(stdin);
    cout << "read mnist data finished" << endl;
}

void data_normal(vector<vector<double>>& X) {
    for (int i=0; i<(int)X.size(); ++i) {
        for (int j=0; j<(int)X[0].size(); ++j) {
            X[i][j] /= 255.0;
        }
    }
}
