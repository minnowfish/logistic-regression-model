import numpy as np
import matplotlib.pyplot as plt

# maps a real value into a value between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    z = np.dot(X, weights) # get raw model predictions
    h = sigmoid(z) 
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    costs = []

    for i in range(iterations):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient

        cost = compute_cost(X, y, weights)
        costs.append(cost)

    return weights, costs

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def fit(self, X, y):
        # get best weight using gradient descent
        X = np.c_[np.ones((X.shape[0], 1)), X] #adds bias to left of X
        self.weights = np.zeros(X.shape[1])
        
        self.weights, self.costs = gradient_descent(X, y, self.weights, self.learning_rate, self.iterations)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X] 
        return sigmoid(np.dot(X, self.weights)) >= 0.5

def plot_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolor='b')

    x_values = [np.min(X[:, 0] - 1), np.max(X[:, 0] + 1)]
    y_values = -(model.weights[0] + np.dot(model.weights[1], x_values)) / model.weights[2]
    plt.plot(x_values, y_values, label="Decision Boundary")

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.show()

# training and testing

# dummy dataset
X_train = np.array([[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression(learning_rate=0.1, iterations=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_train)
print("Predictions:", predictions)

plot_decision_boundary(X_train, y_train, model)

plt.plot(range(model.iterations), model.costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost reduction over time")
plt.show()
