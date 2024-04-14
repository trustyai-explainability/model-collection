import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load

X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=0, class_sep=1.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

path = 'credit-score/model.joblib'

# Save the model to disk
dump(model, path)


model_loaded = load(path)

# Verify the loaded model by using it to make predictions
y_pred_loaded = model_loaded.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print("Loaded Model Accuracy:", accuracy_loaded)
