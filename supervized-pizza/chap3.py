import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict(X, w, b):
    """Make predictions based on input data and weights"""
    return X * w + b


def loss(X, Y, w, b):
    """Calculate the mean squared error loss"""
    error = predict(X, w, b) - Y
    squared_error = error**2  # the error always must be positive
    return np.average(squared_error)


def gradient(X, Y, w):
    return 2 * np.average(X * (predict(X, w, 0) - Y))


def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, 0)
        print(f"Iteration {i:4d} => Loss: {current_loss:.6f}")

        w -= gradient(X, Y, w) * lr
    return w


X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

w = train(X, Y, iterations=100, lr=0.001)
print(f"w={w:.10f}")
