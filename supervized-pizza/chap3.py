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


def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)


def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, 0)
        print(f"Iteration {i:4d} => Loss: {current_loss:.6f}")

        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr

    return w, b


X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

w, b = train(X, Y, iterations=20000, lr=0.001)
print(f"w={w:.10f}, b={b:10f}")

x = 20
y = predict(x, w, b)
print(f"Prediction: x={x} => y={y:.2f}")
