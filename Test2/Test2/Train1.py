import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Flatten, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
from scipy.stats import entropy as scipy_entropy
import random
import string

# Entropy calculation function
def calculate_entropy(text):
    # Calculate the frequency of each character in the text
    char_counts = np.array([float(text.count(c)) for c in set(text)])
    probabilities = char_counts / np.sum(char_counts)
    return scipy_entropy(probabilities)

# Feature extraction from ciphertexts
def extract_features(ciphertexts):
    features = []
    for text in ciphertexts:
        feature_vector = []
        # Example features (you can expand this list)
        entropy = calculate_entropy(text)
        length = len(text)
        unique_chars = len(set(text))
        feature_vector.extend([entropy, length, unique_chars])
        features.append(feature_vector)
    return np.array(features, dtype=np.float32)

# Function to generate random ciphertext (for demonstration purposes)
def generate_ciphertext(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Generate labeled dataset (for demonstration purposes)
def create_dataset():
    data = []
    labels = []
    
    # Simulate ciphertexts for each algorithm
    for algorithm in ['AES', 'DES', 'RSA', 'Blowfish', 'RC4', 'ChaCha20']:
        for _ in range(1000):  # Generate 1000 samples for each algorithm
            if algorithm == 'AES':
                text = generate_ciphertext(128)
            elif algorithm == 'DES':
                text = generate_ciphertext(64)
            elif algorithm == 'RSA':
                text = generate_ciphertext(256)
            elif algorithm == 'Blowfish':
                text = generate_ciphertext(64)
            elif algorithm == 'RC4':
                text = generate_ciphertext(128)
            elif algorithm == 'ChaCha20':
                text = generate_ciphertext(128)
            
            data.append(text)
            labels.append(algorithm)
    
    return pd.DataFrame({'ciphertext': data, 'algorithm': labels})

# Build the CNN-GRU hybrid model
def build_model(input_shape):
    model = Sequential()
    
    # CNN layers
    model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=(input_shape[1], input_shape[2])))  
    model.add(Flatten())

    # Reshape the data before feeding it into GRU
    model.add(Reshape((1, -1)))  # Reshape the flattened output for GRU
    
    # GRU layer
    model.add(GRU(128, return_sequences=False))
    
    # Fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))  # 6 classes for AES, DES, RSA, Blowfish, RC4, ChaCha20
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Step 1: Create and save dataset (you can skip this if you already have a dataset)
    dataset = create_dataset()
    dataset.to_csv('cipher_data.csv', index=False)

    # Step 2: Load data
    data = pd.read_csv('cipher_data.csv')
    ciphertexts = data['ciphertext'].values
    algorithms = data['algorithm'].values

    # Step 3: Extract features
    features = extract_features(ciphertexts)
    
    # Step 4: Encode labels
    encoder = LabelEncoder()
    algorithm_labels = encoder.fit_transform(algorithms)

    # Step 5: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, algorithm_labels, test_size=0.2, random_state=42)

    # Step 6: Reshape input for Conv1D layer (samples, time steps, features)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Step 7: Build and train the model
    model = build_model(X_train.shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Step 8: Save the model and the label encoder
    model.save('cnn_gru_model.h5')
    joblib.dump(encoder, 'label_encoder.pkl')

    # Step 9: Evaluate the model on test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

if __name__ == "__main__":
    main()
