import numpy as np
import pandas as pd
import math
import random
import string
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization

# Entropy calculation function
def calculate_entropy(text):
    probabilities = [float(text.count(c)) / len(text) for c in set(text)]
    return scipy_entropy(probabilities)

# Feature extraction from ciphertexts
def extract_features(ciphertext):
    features = []
    
    for text in ciphertext:
        feature_vector = []
        
        # AES: Entropy Level, Block Patterns, Round Transformation
        aes_entropy = calculate_entropy(text)
        aes_block_patterns = len(set(text))  # Example: unique block patterns
        aes_round_transformation = len(text) // 16  # Example: divide into 16-byte blocks
        
        feature_vector.extend([aes_entropy, aes_block_patterns, aes_round_transformation])

        # DES: Block Size, Feistel Structure, Entropy Level
        des_block_size = len(text)
        des_feistel_structure = des_block_size // 8  # Simplified Feistel round assumption
        des_entropy = calculate_entropy(text)
        
        feature_vector.extend([des_block_size, des_feistel_structure, des_entropy])

        # RSA: Ciphertext Size, Encryption Pattern, Public/Private Key Pair, Low Entropy
        rsa_ciphertext_size = len(text)
        rsa_encryption_pattern = calculate_entropy(text)  # Simplified entropy pattern
        rsa_key_pair = rsa_ciphertext_size % 2  # Simulate detection of key pair (odd/even size)
        rsa_low_entropy = rsa_encryption_pattern < 0.5  # Example threshold for low entropy
        
        feature_vector.extend([rsa_ciphertext_size, rsa_encryption_pattern, rsa_key_pair, rsa_low_entropy])

        # Blowfish: Feistel Rounds, Block Patterns
        blowfish_feistel_rounds = len(text) // 8  # Simulated Feistel rounds
        blowfish_block_patterns = len(set(text))  # Example: unique block patterns
        
        feature_vector.extend([blowfish_feistel_rounds, blowfish_block_patterns])

        # RC4: Keystream Generation, Entropy
        rc4_keystream_generation = len(set(text)) / len(text)  # Simulated keystream ratio
        rc4_entropy = calculate_entropy(text)
        
        feature_vector.extend([rc4_keystream_generation, rc4_entropy])

        # ChaCha20: Stream Cipher, Ciphertext Properties
        chacha_stream_cipher = 1  # Simplified representation of stream cipher
        chacha_properties = len(text) // 8  # Ciphertext length / 8
        
        feature_vector.extend([chacha_stream_cipher, chacha_properties])
        
        features.append(feature_vector)
    
    return np.array(features)

# Function to generate random ciphertext
def generate_ciphertext(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Generate labeled dataset
def create_dataset():
    data = []
    labels = []
    
    # Simulate ciphertexts for each algorithm
    for algorithm in ['AES', 'DES', 'RSA', 'Blowfish', 'RC4', 'ChaCha20']:
        for _ in range(10000):  # Generate 1000 samples for each algorithm
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

# CNN-GRU model
def build_model(input_shape):
    model = Sequential()

    # CNN for feature extraction
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # GRU for sequence processing
    model.add(GRU(128, return_sequences=False))  # Output: (batch_size, 128)
    model.add(Dropout(0.3))

    # Fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))

    # Output layer for algorithm classification
    model.add(Dense(6, activation='softmax'))  # Assuming 6 algorithms (AES, DES, RSA, Blowfish, RC4, ChaCha20)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset
def load_data(filepath):
    # Assumes CSV with 'ciphertext' and 'algorithm' columns
    data = pd.read_csv(filepath)
    ciphers = data['ciphertext'].values
    algorithms = data['algorithm'].values
    return ciphers, algorithms

# Main function
def main():
    # Step 1: Create and save dataset
    dataset = create_dataset()
    dataset.to_csv('cipher_data.csv', index=False)

    # Step 2: Load data
    ciphertexts, algorithms = load_data('cipher_data.csv')

    # Step 3: Extract features using real calculations
    features = extract_features(ciphertexts)

    # Step 4: Encode algorithm labels
    encoder = LabelEncoder()
    algorithm_labels = encoder.fit_transform(algorithms)

    # Step 5: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, algorithm_labels, test_size=0.2, random_state=42)

    # Reshape data for CNN input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Step 6: Build and train the CNN-GRU model
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    # Step 7: Evaluate model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, target_names=encoder.classes_))

if __name__ == '__main__':
    main()
