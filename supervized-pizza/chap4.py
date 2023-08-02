import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)

print(x1.shape)

X = np.column_stack((x1, x2, x3))
# y is in a one dimensional form (30,) instead (30,1), and mixing arrays with matrix could result in unexpected behavior
Y = y.reshape(-1, 1)

# we initialize w to a matrix of zeros, the number of columns in the input matrix and 1 column for each weight value
w = np.zeros((X.shape[1], 1))


def predict(X, w):
    return np.matmul(X, w)


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def gradient(X, Y, w):
    x_transposed = X.T
    return 2 * np.matmul(x_transposed, (predict(X, w) - Y)) / X.shape[0]


print(X.shape)
