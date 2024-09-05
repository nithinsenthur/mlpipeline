import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os

# Load the model from the .keras file
model = tf.keras.models.load_model('model.keras')

# Get the tracking URI
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

# Step 2: Log the model into MLflow
with mlflow.start_run():

    # Log model architecture parameters
    mlflow.log_param("layer_1_units", 64)
    mlflow.log_param("layer_1_activation", 'relu')
    mlflow.log_param("layer_2_units", 32)
    mlflow.log_param("layer_2_activation", 'relu')
    mlflow.log_param("output_layer_units", 1)
    mlflow.log_param("output_layer_activation", 'sigmoid')

    # Log training parameters
    mlflow.log_param("optimizer", 'adam')
    mlflow.log_param("loss_function", 'binary_crossentropy')
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)

    # Log hyperparameters
    mlflow.log_param("learning_rate", 0.001)  # Default learning rate for Adam optimizer

    # Log the TensorFlow model to MLflow
    mlflow.tensorflow.log_model(model, "model_v1")