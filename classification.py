# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Load Iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Calculate precision
precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

# Calculate recall
recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

# Calculate F1-score
f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

# Display results
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)