import numpy as np
import matplotlib.pyplot as plt

data_train = np.loadtxt('train.txt', delimiter=',')#считываем кучу массивов
data_test = np.loadtxt('test.txt', delimiter=',')

X_train = data_train[:, 0]
y_train = data_train[:, 1]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)  # ?

X_test = data_test[:, 0]
y_test = data_test[:, 1]

X_test = X_test.reshape(-1, 1)


class LinearRegression:
    """
    k * x + b
    """

    def __init__(self):
        self.koeffs = None

    def fit(self, X, y):
        X = np.hstack((X, X ** 0))#типа стыкуем
        X_cross = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
        self.koeffs = np.matmul(X_cross, y).T[0]

    def predict(self, x):
        y_pred = self.koeffs[0] * x + self.koeffs[1] * 1
        return y_pred

    def print_koeffs(self):
        print(self.koeffs[0], "x + ", self.koeffs[1], sep="")


def plot(x, y, x_prediction, y_prediction):
    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.plot(x_prediction, y_prediction)

    plt.show()


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_prediction = [lr.predict(x) for x in X_train.T[0]]
lr.print_koeffs()

plot(X_train, y_train, X_train, lr_prediction)
