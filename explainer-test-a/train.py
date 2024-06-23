import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(42)
n_samples = 1000

# Feature 1: High impact on output
X1 = np.random.normal(0, 1, n_samples)
# Feature 2: Low impact on output
X2 = np.random.normal(0, 1, n_samples)

# Output: Linear combination of features with added noise
y = 10 * X1 + 0.1 * X2 + np.random.normal(0, 1, n_samples)

X = pd.DataFrame({'X1': X1, 'X2': X2})
y = pd.Series(y, name='y')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("Model Coefficients:")
print(coefficients)

# Predictions
y_pred = model.predict(X_test)

joblib_file = "model.joblib"
joblib.dump(model, joblib_file)

print(f"Model saved as {joblib_file}.")
