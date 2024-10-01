import numpy as np
import random
import pickle
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GRU, Dense
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model

# Cipher generation functions (dummy implementations)
def generate_amsco_cipher(plaintext):
    return ''.join(random.sample(plaintext, len(plaintext)))

def generate_autokey_cipher(plaintext, key='KEY'):
    ciphertext = []
    key_stream = key + plaintext
    for i in range(len(plaintext)):
        shift = ord(key_stream[i]) - ord('A')
        ciphertext.append(chr(((ord(plaintext[i]) - ord('A') + shift) % 26) + ord('A')))
    return ''.join(ciphertext)

def generate_baconian_cipher(plaintext):
    bacon_dict = {'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA'}
    return ''.join([bacon_dict.get(c, 'AAAAA') for c in plaintext])

def generate_bazeries_cipher(plaintext, key='KEY'):
    return ''.join(random.sample(plaintext, len(plaintext)))

def generate_beaufort_cipher(plaintext, key='KEY'):
    ciphertext = []
    for i in range(len(plaintext)):
        shift = ord(key[i % len(key)]) - ord('A')
        ciphertext.append(chr(((ord('Z') - ord(plaintext[i]) + shift) % 26) + ord('A')))
    return ''.join(ciphertext)

# Function to generate dataset for training
def generate_dataset(size):
    data = []
    labels = []
    algorithms = ['amsco', 'autokey', 'baconian', 'bazeries', 'beaufort']
    plaintext = "HELLOWORLD"  # Example plaintext, can vary

    for _ in range(size):
        algo = random.choice(algorithms)
        if algo == 'amsco':
            cipher = generate_amsco_cipher(plaintext)
        elif algo == 'autokey':
            cipher = generate_autokey_cipher(plaintext)
        elif algo == 'baconian':
            cipher = generate_baconian_cipher(plaintext)
        elif algo == 'bazeries':
            cipher = generate_bazeries_cipher(plaintext)
        elif algo == 'beaufort':
            cipher = generate_beaufort_cipher(plaintext)

        data.append([ord(c) for c in cipher])  # Convert characters to ASCII codes
        labels.append(algorithms.index(algo))  # Label the algorithm

    return data, labels

# Data preprocessing
def preprocess_data(data, labels):
    # Padding the sequences to ensure uniform length
    padded_data = pad_sequences(data, padding='post')  # Post padding with zeros

    # Reshape data to be compatible with CNN input
    padded_data = padded_data.reshape((padded_data.shape[0], padded_data.shape[1], 1))

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=5)

    return padded_data, labels

# CNN-GRU Model
def create_cnn_gru_model(input_shape):
    model = Sequential()

    # Feature extraction with CNN
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # Pass the CNN output directly to the GRU without flattening
    model.add(GRU(units=64, activation='relu', return_sequences=False))

    # Dense output layer with softmax for multiclass classification
    model.add(Dense(5, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Save the model and encoders
def save_model_and_encoders(model, label_encoder, model_filename, encoder_filename):
    # Save the trained model
    model.save(model_filename)
    print(f"Model saved as {model_filename}")

    # Save the label encoder
    with open(encoder_filename, 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)
    print(f"Label encoder saved as {encoder_filename}")

# Load model and encoders for prediction
def load_model_and_encoders(model_filename, encoder_filename):
    # Load the trained model
    model = load_model(model_filename)
    print(f"Model loaded from {model_filename}")

    # Load the label encoder
    with open(encoder_filename, 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    print(f"Label encoder loaded from {encoder_filename}")
    
    return model, label_encoder

# Main function
if __name__ == "__main__":
    # Generate dataset
    data, labels = generate_dataset(size=10000)

    # Preprocess data
    data, labels = preprocess_data(data, labels)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_cnn_gru_model(input_shape=(X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save the model and label encoders
    label_encoder = ['amsco', 'autokey', 'baconian', 'bazeries', 'beaufort']  # Manually creating label encoder
    save_model_and_encoders(model, label_encoder, "cipher_model.h5", "label_encoder.pkl")

    # To load the model and encoder, use this:
    # loaded_model, loaded_encoder = load_model_and_encoders("cipher_model.h5", "label_encoder.pkl")
