#include "func.h"

void read_mnist(vector<vector<double>>& X, vector<vector<double>>& Y, string filename) {
    assert(X.size()==Y.size());
    int num = X.size();

    ifstream fin(filename);

    for (int i=0; i<num; ++i) {
        string line;
        getline(fin, line);
        istringstream lin(line);
        string str;
        int j = 0;
        while (getline(lin, str, ',')) {
            double val = 0;
            stringstream ss(str);
            ss >> val;
            if (!j) {
                Y[i][val] = 1;
                j++;
            } else {
                X[i][j-1] = val;
                j++;
            }
        }
    }
    fin.close();
    cout << "read mnist data finished" << endl;
}

void data_normal(vector<vector<double>>& X) {
    for (int i=0; i<(int)X.size(); ++i) {
        for (int j=0; j<(int)X[0].size(); ++j) {
            X[i][j] /= 255.0;
        }
    }
}
