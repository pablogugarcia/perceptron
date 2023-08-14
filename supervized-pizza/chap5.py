import numpy as np


# logistic sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# we used to calculate the weighted sum with the following fn:
# def predict(X, w):
#     return np.matmul(X , w)
# forward pass the calculation through sigmoid fn (forward propagation)
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)


def classify(X, w):
    return np.round(forward(X, w))


# mean squared error loss
def mse_loss(X, Y, w):
    return np.average((forward(X, w) - Y) ** 2)


def log_loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]
