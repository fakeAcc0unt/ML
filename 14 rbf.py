import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (X, y) for a simple function approximation task
def generate_data(num_points=100):
    X = np.linspace(-5, 5, num_points)
    y = np.sin(X) + 0.1 * np.random.randn(num_points)  # Sinusoidal data with noise
    return X.reshape(-1, 1), y.reshape(-1, 1)

# RBF Network Class
class RBFNetwork:
    def _init_(self, num_centers, input_dim, sigma=1.0):
        self.num_centers = num_centers
        self.input_dim = input_dim
        self.sigma = sigma
        
        # Randomly initialize centers
        self.centers = np.random.uniform(-5, 5, (num_centers, input_dim))
        
        # Initialize weights
        self.weights = np.random.rand(num_centers, 1)

    def rbf(self, x, center):
        return np.exp(-np.linalg.norm(x - center)*2 / (2 * self.sigma*2))

    def _calculate_rbf_output(self, X):
        rbf_outputs = np.zeros((X.shape[0], self.num_centers))
        for i, center in enumerate(self.centers):
            rbf_outputs[:, i] = np.apply_along_axis(self.rbf, 1, X, center)
        return rbf_outputs

    def fit(self, X, y):
        rbf_outputs = self._calculate_rbf_output(X)
        # Solve for weights using least squares
        self.weights = np.linalg.pinv(rbf_outputs).dot(y)

    def predict(self, X):
        rbf_outputs = self._calculate_rbf_output(X)
        return rbf_outputs.dot(self.weights)

# Generate data
X_train, y_train = generate_data()

# Create and train the RBF network
num_centers = 10  # Number of RBF centers
rbf_network = RBFNetwork(num_centers=num_centers, input_dim=1, sigma=1.0)
rbf_network.fit(X_train, y_train)

# Generate test data for prediction
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
y_pred = rbf_network.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='red', label='Training Data', s=10)
plt.plot(X_test, y_pred, color='blue', label='RBF Network Prediction')
plt.title('Radial Basis Function Network')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
