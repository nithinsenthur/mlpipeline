import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import ast

# Configure TensorFlow to use multiple GPUs
strategy = tf.distribute.MirroredStrategy()

# Read the processed data
df = pd.read_csv("data.csv")

# Remove the rows where the scaled features don't have 10 elements
def is_valid_length(scaled_feature_str):
    parsed_list = ast.literal_eval(scaled_feature_str)
    return len(parsed_list) == 10

df_filtered = df[df['scaled_features'].apply(is_valid_length)]

# Separate the features and the label set
X = np.array([ast.literal_eval(item) for item in df_filtered['scaled_features']])
y = np.array(df_filtered['Label'].tolist())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Use the MirroredStrategy scope to create and compile the model
with strategy.scope():
    # Simple feedforward neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Optional: Save the model
model.save("model.keras")