import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from scipy.stats import entropy as scipy_entropy

# Entropy calculation function
def calculate_entropy(text):
    probabilities = [float(text.count(c)) / len(text) for c in set(text)]
    return scipy_entropy(probabilities)

# Feature extraction function (same as the one used during training)
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

# Load the trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Main evaluation function (accepting user input)
def evaluate(model_path, label_encoder_path):
    # Step 1: Load the trained model
    model = load_trained_model(model_path)

    # Step 2: Get ciphertexts from user input
    ciphertexts = []
    print("Enter the ciphertexts (enter 'exit' to stop):")
    while True:
        user_input = input("Ciphertext: ")
        if user_input.lower() == 'exit':
            break
        ciphertexts.append(user_input)

    if len(ciphertexts) == 0:
        print("No ciphertexts provided. Exiting.")
        return
    
    print("Ciphertexts loaded:", ciphertexts)  # Debug print

    # Step 3: Extract features from the user-provided ciphertexts
    features = extract_features(ciphertexts)
    print("Features shape before reshaping:", features.shape)  # Debug print

    # Check if features were extracted
    if features.size == 0:
        print("No features extracted. Check the input data.")
        return

    # Step 4: Reshape data for CNN input
    features = features.reshape((features.shape[0], features.shape[1], 1))

    # Step 5: Make predictions
    predictions = model.predict(features)
    predicted_classes = np.argmax(predictions, axis=1)

    # Step 6: Load the Label Encoder to decode class labels
    encoder = LabelEncoder()
    encoder.classes_ = np.load(label_encoder_path)  # Assuming you saved the classes with np.save
    predicted_algorithms = encoder.inverse_transform(predicted_classes)

    # Step 7: Print the predictions
    for i, cipher in enumerate(ciphertexts):
        print(f"Ciphertext: {cipher} -> Predicted Algorithm: {predicted_algorithms[i]}")

if __name__ == '__main__':
    model_path = 'cnn_gru_model.h5'  # Path to the trained model
    label_encoder_path = 'label_encoder_classes.npy'  # Path to the saved LabelEncoder classes
    
    evaluate(model_path, label_encoder_path)
