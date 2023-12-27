import numpy as np
import gzip
import struct


def load_images(filename):
    # open an unzip the file of images
    with gzip.open(filename, "rb") as f:
        # Read the header information into a bunch of variables
        _ignored, n_images, columns, rows = struct.unpack(">IIII", f.read(16))
        # Read all the pixel into a numpy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)


def prepend_bias(X):
    # Insert a column of 1s in the position 0 of X.​
    # (“axis=1” stands for: “insert a column, not a row”)
    return np.insert(X, 0, 1, axis=1)


# 60000 images,each 785 elements (1 bias + 28 * 28 pixels)
X_train = prepend_bias(
    load_images("../book_source_code/data/mnist/train-images-idx3-ubyte.gz")
)

# 10000 images, same structure
X_test = prepend_bias(
    load_images("../book_source_code/data/mnist/t10k-images-idx3-ubyte.gz")
)


def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        f.read(8)
        all_labels = f.read()
        # reshape the labels into a one label matrix
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def encode_fives(Y):
    # Converts all 5 to 1, and everything else to 0
    return (Y == 5).astype(int)


def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y


# Y_train = encode_fives(
#     load_labels("../book_source_code/data/mnist/train-labels-idx1-ubyte.gz")
# )

# Y_test = encode_fives(
#     load_labels("../book_source_code/data/mnist/t10k-labels-idx1-ubyte.gz")
# )
Y_train_unencoded = load_labels(
    "../book_source_code/data/mnist/train-labels-idx1-ubyte.gz"
)

Y_train = one_hot_encode(Y_train_unencoded)

Y_test = load_labels("../book_source_code/data/mnist/t10k-labels-idx1-ubyte.gz")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)


def classify(X, w):
    # matrix of predictions yhat one row per label one column per class
    y_hat = forward(X, w)
    # then argmax get the maximum value of each row of yhat, the result is an array of indices
    labels = np.argmax(y_hat, axis=1)
    # reshap the labels to a single column matrix
    return labels.reshape(-1, 1)


def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.sum(first_term + second_term) / X.shape[0]


def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]


def train(X_train, Y_train, X_test, Y_test, iterations, lr):
    w = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for i in range(iterations):
        report(i, X_train, Y_train, X_test, Y_test, w)
        w -= gradient(X_train, Y_train, w) * lr
    report(iterations, X_train, Y_train, X_test, Y_test, w)
    return w


def report(iteration, X_train, Y_train, X_test, Y_test, w):
    matches = np.count_nonzero(classify(X_test, w) == Y_test)
    n_test_examples = Y_test.shape[0]
    matches = matches * 100.0 / n_test_examples
    training_loss = loss(X_train, Y_train, w)
    print("%d - Loss: %.20f, %.2f%%" % (iteration, training_loss, matches))


def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print(
        "\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent)
    )


w = train(X_train, Y_train, X_test, Y_test, iterations=200, lr=1e-5)
