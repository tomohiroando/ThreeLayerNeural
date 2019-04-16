import numpy as np
import math
import matplotlib.pyplot as plt

def make_data(num, scope):
    """
    データセット作成
    num: 個数
    scope: 値の範囲(1-scope)
    """
    X = np.random.rand(num, 2) * scope
    T = np.array([(1.0 + math.sin(4.0 * math.pi * x[0])) * x[1] / 2.0 for x in X])
    T = 1 / (1 + np.exp(-T))

    return X, T.reshape(-1, 1)

class Neural:
    def __init__(self, n_input, n_hidden, n_output):
        """
        ネットワーク初期化
        n_input: 入力次元数
        n_hidden: 隠れ層の次元数
        n_output: 出力次元数
        """
        self.W1 = np.random.random_sample((n_input, n_hidden))
        self.W2 = np.random.random_sample((n_hidden, n_output))
        self.b1 = np.random.random_sample(n_hidden)
        self.b2 = np.random.random_sample(n_output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def forward(self, x):
        Z = self.sigmoid(np.dot(x, self.W1) + self.b1)
        Y = self.sigmoid(np.dot(Z, self.W2) + self.b2)

        return Z, Y

    def backward(self, x, t, eta):
        """
        逆伝播
        x: 入力データ
        t: 正解データ
        eta: 学習率
        """
        # 順伝播によって出力値を出力
        z, y = self.forward(x)

        # 中間層→出力層の重み更新
        h_o_delta = (t - y) * (y * (1.0 - y))
        self.W2 += (eta * h_o_delta.reshape(-1, 1) * z).T

        # バイアスの更新
        self.b2 += (eta * h_o_delta.T * np.array([1.0])).T

        # 入力層→中間層の重み更新
        i_h_delta = np.dot(self.W2, h_o_delta).T * z * (1.0 - z)
        self.W1 += (eta * i_h_delta.reshape(-1, 1) * x).T
        # バイアスの更新
        self.b1 += (eta * i_h_delta.T * np.array([1.0])).T

    def loss_calculation(self, X, T):
        # 二乗誤差を求める
        loss = 0.0
        for x, t in zip(X, T):
            z, y = self.forward(x)
            loss += np.dot((y - t), ((y - t).reshape(-1, 1))) / 2.0 

        return loss


    def train(self, X, T, eta, EPOCH_NUM):
        """
        逆伝播による損失の算出
        """
        self.loss = np.zeros(EPOCH_NUM)
        #shape[0]は行数
        #shape[1]は列数
        for epoch in range(EPOCH_NUM):
            for x, t in zip(X, T):
                self.backward(x, t, eta)

            self.loss[epoch] = self.loss_calculation(X,T)

    def predict(self, X):
        Y = np.zeros((X.shape[0], X.shape[1]))
        for i, x in enumerate(X):
            z, y = self.forward(x)
            Y[i] = y

        return Y

    def plt_loss(self):
         plt.plot(np.arange(0, self.loss.shape[0]), self.loss)
         plt.show()



if __name__ == '__main__':
    X, T = make_data(100, 100)

    input_size = X.shape[1]
    hidden_size = 3
    output_size = 1
    eta = 0.1
    EPOCH_NUM = 10000

    nn = Neural(input_size, hidden_size, output_size)
    nn.train(X, T, eta, EPOCH_NUM)
    nn.plt_loss()








