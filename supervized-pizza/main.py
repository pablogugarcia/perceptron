import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Set plot limits and font sizes
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel('Reservations', fontsize=30)
plt.ylabel('Pizzas', fontsize=30)


# Load data from file
X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True) # load data from file

# plt.plot(X, Y, 'bo')
# plt.show() # -> print the chart

def predict(X , w) :
    """Make predictions based on input data and weights"""
    return X * w


def loss(X, Y, w):
    """Calculate the mean squared error loss"""
    error = predict(X, w) - Y
    squared_error = error ** 2 # the error always must be positive
    return np.average(squared_error)


def train (X, Y ,iterations, lr ): # lr is the short form for learning rate
    """Train the model using gradient descent"""
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print(f"Iteration {i:4d} => Loss: {current_loss:.6f}")

        if loss(X,Y , w +lr ) <current_loss:
            w += lr
        elif loss(X, Y , w - lr) < current_loss:
            w -= lr
        else:
            return w
    
    raise Exception(f"Couldn't converge within {iterations} iterations")




# Train the system
w = train(X,Y, iterations=10000, lr= 0.01)
print(f"\nw = {w:.3f}")

# Predict the number of pizzas
x = 20
prediction = predict(x, w)
print(f"Prediction: x={x} => y={prediction:.2f}")
