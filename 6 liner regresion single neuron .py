import numpy as np

# Generate some example data (simple linear relationship y = 2x + 1)
np.random.seed(42)
X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # y = 2x + 1 + some noise

# Initialize weight (w) and bias (b) randomly
w = np.random.randn(1)
b = np.random.randn(1)

# Hyperparameters
learning_rate = 0.1
epochs = 1000

# Linear regression function
def linear_regression(X, w, b):
    return X.dot(w) + b

# Loss function (Mean Squared Error)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Training loop (Gradient Descent)
for epoch in range(epochs):
    # Forward pass: Compute predictions
    y_pred = linear_regression(X, w, b)
    
    # Compute the loss
    loss = mse_loss(y, y_pred)
    
    # Backward pass: Compute gradients
    dw = -2 * np.mean((y - y_pred) * X)
    db = -2 * np.mean(y - y_pred)
    
    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {loss}')

print(f'Final weight: {w}, Final bias: {b}')
