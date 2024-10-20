import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
# Load the dataset
data = pd.read_csv('/content/Housing.csv')
# Inspect the first few rows of the dataset
print(data.head())
# Check for missing values
print(data.isnull().sum())
# Handle missing values if necessary (e.g., fill with mean, median, or
drop)
data = data.dropna()
# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)
# Explore the data with some visualizations
sns.pairplot(data)
plt.show()
# Define the features (X) and the target (y)
X = data.drop('price', axis=1) # Assuming 'PRICE' is the column name for
house prices
y = data['price']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
# Initialize the Linear Regression model
model = LinearRegression()
# Train the model on the training data
model.fit(X_train, y_train)
# Print the coefficients of the model
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
# Make predictions on the testing data
y_pred = model.predict(X_test)
# Calculate the Mean Squared Error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared value: {r2}")
# Plot the predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
