import serial
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("gesture_recognition_model.keras")

# Parameters
time_steps = 10  # Same as used during training
features = 1     # Single feature per time step

# Function to convert hex string to integer array
def hex_to_int(sequence):
    try:
        return [int(val, 16) for val in sequence.split()]
    except ValueError:
        print("Error: Invalid hex data received")
        return []

# Function to pad and reshape data for the model
def pad_and_reshape(sequence, time_steps, features):
    padded_sequence = np.pad(sequence, (0, max(0, time_steps - len(sequence))), 'constant')[:time_steps]
    return np.array(padded_sequence).reshape((1, time_steps, features))

# Open serial port
try:
    ser = serial.Serial(port='COM3', baudrate=9600, timeout=1)  # Update 'COM3' to match your port
    print("Serial port opened successfully.")
except serial.SerialException as e:
    print(f"Error: Could not open serial port - {e}")
    exit()

print("Listening for data...")

try:
    while True:
        # Read a line of data from the serial port
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue

        print(f"Received: {line}")

        # Convert the data from hex string to integer array
        sequence = hex_to_int(line)
        if not sequence:
            continue

        # Prepare the data for the model
        input_data = pad_and_reshape(sequence, time_steps, features)

        # Make a prediction
        prediction = model.predict(input_data)
        label = "hi" if prediction[0][0] < 0.5 else "clapping"
        print(f"Prediction: {label} (Confidence: {prediction[0][0]:.2f})")

except KeyboardInterrupt:
    print("Program interrupted by user. Exiting...")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if ser.is_open:
        ser.close()
    print("Serial port closed.")
