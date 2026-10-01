# Import required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# House area in square feet
X = np.array([
    [1000],
    [1200],
    [1500],
    [1800],
    [2000],
    [2200],
    [2500],
    [2800],
    [3000],
    [3500]
])

# House prices in lakhs
y = np.array([
    30,
    36,
    45,
    54,
    60,
    66,
    75,
    84,
    90,
    105
])

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-Squared
r2 = r2_score(y_test, y_pred)

# Display actual and predicted values
print("Actual Prices:", y_test)
print("Predicted Prices:", y_pred)

# Display performance metrics
print("\nMean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-Squared (R2):", r2)