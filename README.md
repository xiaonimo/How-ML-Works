# Multi-Layer BP Neural Network

## Introductions
- ### data
  - download from [kaggle](https://www.kaggle.com/c/3004/download/train.csv)
  - you can read mnist data like this:
  ```
    vector<vector<double>> X(40000, vector<double>(784, 0));
    vector<vector<double>> Y(40000, vector<double>(10, 0));
    read_mnist(X, Y, "train.csv");
  ```
  - normalized data like this:
  ```
    data_normal(X);
  ```
- ### how to train
  - you can use like this:
  ```
    BP a(vector<int>{784, 100, 10});
    a.set_train_data(X, Y, 0.9);
    a.train();
    a.predict();
  ```
- ### how to save and load model
  - if there is a trained-well model, you can use like this:
  ```
    BP a;
    a.load_model("model_36000.txt");
    a.set_train_data(X, Y, 0);
    a.predict();
  ```
  - if you want to save the model, you can use like this:
  ```
  a.save_model("model_36000.txt");
  ```

- ### what's in the model file?
  - from input layer to output layer, how may neurals in each layer
  - activation(sigmoid ro relu)
  - Weights
  - Bias

## Change Log
### 2018.3.23
- [x] BP()
- [x] index()
- [x] forword_flow()
- [x] activation()
- [x] backword_flow()
- [x] train()
- [x] predict()
- [x] data_normalization()

### 2018.3.24
- [ ] improve the accuracy while using Mini-BGD
- [ ] L2 regularization

### 2018.3.26
- [x] 36000 items to train, 4000 items to test, only one hidden layer with 100 neurals, max_itr_all=10, min_loss=0.01, SGD, **accuracy>0.97**

### 2018.3.27
- [x] save_model()
- [x] load_model()
- [x] add a model file("model_36000.txt") which accuracy>**0.99**
