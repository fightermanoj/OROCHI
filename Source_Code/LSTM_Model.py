import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('gesture_recognition_model.h5')

# Example function to preprocess input data and make predictions
def preprocess_input(data, time_steps=10):
    # Convert hex strings to integers (assuming you have the hex_to_int function)
    data_sequence = [list(int(val, 16) for val in data.split()) for data in data]

    # Pad sequences to ensure they are all the same length
    data_padded = pad_sequences(data_sequence, maxlen=time_steps, padding='post', dtype='float32')

    # Reshape data to 3D (samples, time_steps, features)
    data_reshaped = data_padded.reshape((data_padded.shape[0], time_steps, 1))

    return data_reshaped

# Example of using the model for predictions
input_data = "53 59 80 03 00 01 38 68 54 43"  # Example input data (replace with actual data)
processed_data = preprocess_input([input_data])  # Process the data

# Predict the gesture
prediction = model.predict(processed_data)
print("Predicted gesture:", "Clapping" if prediction[0] > 0.5 else "Hi")
