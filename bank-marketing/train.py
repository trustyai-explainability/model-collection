import pandas as pd
from keras import Input
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load data
data = pd.read_csv('./data/train.csv', sep=';', header=0)

X = data.drop('y', axis=1)
y = data['y']

print(data.info())
print(data.head())

# Preprocessing
y = (y == 'yes').astype(int)

numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Create a pipeline for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the training data
X_train_prep = pipeline.fit_transform(X_train)
X_test_prep = pipeline.transform(X_test)

# Create a simple neural network
model = Sequential([
    Input(shape=(X_train_prep.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_prep, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_prep, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save model
tf.keras.models.save_model(model, "model.h5")
print("Model saved successfully.")