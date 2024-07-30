# train_model.py
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

# Fetch the dataset
housing = fetch_california_housing(return_X_y=True)
X, y = housing

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42
)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model to a file
model_file = "model.joblib"
joblib.dump(model, model_file)

print(f"Model saved to {model_file}")
