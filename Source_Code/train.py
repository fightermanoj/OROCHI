import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import time

# 1. Load the CSV file
df = pd.read_csv('gestures.csv')

# Convert sequences from hex strings to integer arrays
def hex_to_int(sequence):
    return [int(val, 16) for val in sequence.split()]

df['sequence'] = df['sequence'].apply(hex_to_int)

# Convert to features and labels
X = df['sequence'].values
y = df['label'].map({'hi': 0, 'clapping': 1}).values

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Pad and reshape data
time_steps = 10  # Each sequence has 10 values
features = 1     # Each value is a single feature

# Pad each sequence to have a length of time_steps
def pad_sequences(sequences, time_steps):
    padded = []
    for seq in sequences:
        padded_seq = np.pad(seq, (0, max(0, time_steps - len(seq))), 'constant')[:time_steps]
        padded.append(padded_seq)
    return np.array(padded)

# Pad and reshape the training and testing data
X_train_padded = pad_sequences(X_train, time_steps).reshape((-1, time_steps, features))
X_test_padded = pad_sequences(X_test, time_steps).reshape((-1, time_steps, features))

# Flatten labels (if needed)
y_train = y_train.flatten()
y_test = y_test.flatten()

# Verify shapes
print(f"X_train shape: {X_train_padded.shape}")
print(f"X_test shape: {X_test_padded.shape}")

# 3. Define and compile the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
tensorboard = TensorBoard(log_dir='logs')

model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(time_steps, features)))
model.add(Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Train the model
start_time = time.time()

history = model.fit(X_train_padded, y_train, epochs=100, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping, tensorboard])

end_time = time.time()
print(f"Training took {end_time - start_time:.2f} seconds.")

# 5. Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test accuracy: {accuracy}")

# 6. Save the trained model
model.save("gesture_recognition_model.keras")

# 7. Measure inference time
start_time = time.time()
model.predict(X_test_padded[:10])  # Test on the first 10 samples
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference took {inference_time:.2f} seconds for 10 samples.")

# 8. Visualize training metrics
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
