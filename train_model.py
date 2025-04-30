import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load datasets from text files
neutral_df = pd.read_csv(r"C:\Users\KIIT\Desktop\6 sem\neutral.txt")  
punching_df = pd.read_csv(r"C:\Users\KIIT\Desktop\6 sem\punching.txt")  
slapping_df = pd.read_csv(r"C:\Users\KIIT\Desktop\6 sem\slapping.txt")  

# Prepare data and labels
X = []
y = []
no_of_timesteps = 20  # Matching real-time script

# Function to extract sequences from dataset
def create_sequences(df, label):
    data = df.iloc[:, 1:].values  # Remove first column (if index exists)
    n_samples = len(data)
    for i in range(no_of_timesteps, n_samples):
        X.append(data[i-no_of_timesteps:i, :])
        y.append(label)

# Assign labels (0: neutral, 1: Punching, 2: Slapping)
create_sequences(neutral_df, 0)
create_sequences(punching_df, 1)
create_sequences(slapping_df, 2)

# Convert to numpy arrays
X, y = np.array(X), np.array(y)
y = to_categorical(y, num_classes=3)  # One-hot encode labels

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Define LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")  # 3 classes: neutral, Punching, Slapping
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save Model as "lstm-model.h5"
model.save(r"C:\Users\KIIT\Desktop\6 sem\lstm-model.h5")
print("Model saved successfully as 'lstm-model.h5'")
