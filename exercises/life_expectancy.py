import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict(X, w):
    return np.matmul(X, w)


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w


x1, x2, x3, y = np.loadtxt(
    "life-expectancy-without-country-names.txt", skiprows=1, unpack=True
)
print(x1[1], x2[1], x3[1], y[0])

X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)

w = train(X, Y, iterations=1000000, lr=0.000001)

print("\nweights: %s" % w.T)
print("\nPredictions:")

for i in range(5):
    print("X[%d] -> %.4f (label: %d) " % (i, predict(X[i], w), Y[i]))
