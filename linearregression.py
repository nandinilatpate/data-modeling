import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# House area and price
area = np.array([500, 600, 700, 800, 900, 1000, 1200, 1500, 1800, 2000])
price = np.array([50, 60, 70, 80, 90, 100, 120, 150, 180, 200])

# Convert to 2D
X = area.reshape(-1, 1)
y = price

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Actual prices:", y_test)
print("Predicted prices:", y_pred)

# MAE
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# New house prediction
new_house = np.array([[1300]])
print("Predicted price:", model.predict(new_house)[0])