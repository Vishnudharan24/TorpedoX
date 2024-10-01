import random
import string














from Crypto.Cipher import DES, DES3, AES, Blowfish, ARC4
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from Crypto.Cipher import ChaCha20_Poly1305, Salsa20
import binascii


class CryptoUtils:
    
    def __init__(self):
        self.algorithms = [
            (self.aes_encrypt, 'AES'),
            (self.des_encrypt, 'DES'),
            (self.blowfish_encrypt, 'Blowfish'),
            (self.chacha20_poly1305_encrypt, 'ChaCha20-Poly1305'),
            (self.twofish_encrypt, 'Twofish'),
            (self.idea_encrypt, 'IDEA'),
            (self.rc4_encrypt, 'RC4'),
            (self.salsa20_encrypt, 'Salsa20'),
            (self.triple_des_encrypt, 'Triple DES')
        ]
    
    @staticmethod
    def triple_des_encrypt(message, key=None):
        if key is None:
            key = get_random_bytes(24)  # 3DES key size
        else:
            key = bytes.fromhex(key)
        
        cipher = DES3.new(key, DES3.MODE_CBC)
        iv = cipher.iv
        encrypted_message = cipher.encrypt(pad(message.encode(), DES3.block_size))
        encrypted_message_with_iv = iv + encrypted_message

        formatted_ciphertext = ' '.join(binascii.hexlify(encrypted_message_with_iv).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(encrypted_message_with_iv).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')
        
        return formatted_ciphertext, key_hex

    @staticmethod
    def aes_encrypt(message, key=None):
        if key is None:
            key = get_random_bytes(16)  # AES-128 requires a 16-byte key
        else:
            key = bytes.fromhex(key)
        
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        encrypted_message = cipher.encrypt(pad(message.encode(), AES.block_size))
        encrypted_message_with_iv = iv + encrypted_message

        formatted_ciphertext = ' '.join(binascii.hexlify(encrypted_message_with_iv).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(encrypted_message_with_iv).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')

        return formatted_ciphertext, key_hex

    @staticmethod
    def blowfish_encrypt(message, key=None):
        if key is None:
            key = get_random_bytes(16)  # Blowfish key size
        else:
            key = bytes.fromhex(key)
        
        cipher = Blowfish.new(key, Blowfish.MODE_CBC)
        iv = cipher.iv
        encrypted_message = cipher.encrypt(pad(message.encode(), Blowfish.block_size))
        encrypted_message_with_iv = iv + encrypted_message

        formatted_ciphertext = ' '.join(binascii.hexlify(encrypted_message_with_iv).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(encrypted_message_with_iv).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')
        
        return formatted_ciphertext, key_hex

    @staticmethod
    def chacha20_poly1305_encrypt(message, key=None):
        if key is None:
            key = get_random_bytes(32)  # ChaCha20 key size
        else:
            key = bytes.fromhex(key)
        
        cipher = ChaCha20_Poly1305.new(key=key)
        ciphertext, tag = cipher.encrypt_and_digest(message.encode())

        formatted_ciphertext = ' '.join(binascii.hexlify(ciphertext).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(ciphertext).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')
        
        return formatted_ciphertext, key_hex

    @staticmethod
    def salsa20_encrypt(message, key=None):
        if key is None:
            key = get_random_bytes(32)  # Salsa20 key size
        else:
            key = bytes.fromhex(key)
        
        cipher = Salsa20.new(key=key)
        encrypted_message = cipher.encrypt(message.encode())

        formatted_ciphertext = ' '.join(binascii.hexlify(encrypted_message).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(encrypted_message).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')
        
        return formatted_ciphertext, key_hex


    @staticmethod
    def des_encrypt(message, key=None):
        if key is None:
            key = get_random_bytes(8)  # DES requires an 8-byte key
        else:
            key = bytes.fromhex(key)
        
        cipher = DES.new(key, DES.MODE_CBC)
        iv = cipher.iv
        encrypted_message = cipher.encrypt(pad(message.encode(), DES.block_size))
        encrypted_message_with_iv = iv + encrypted_message

        formatted_ciphertext = ' '.join(binascii.hexlify(encrypted_message_with_iv).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(encrypted_message_with_iv).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')
        
        return formatted_ciphertext, key_hex

    @staticmethod
    def idea_encrypt(message, key=None):
        def xor_bytes(a, b):
            return bytes(x ^ y for x, y in zip(a, b))

        def idea_encrypt_block(plaintext, key):
            return xor_bytes(plaintext, key)

        if key is None:
            key = get_random_bytes(16)  # IDEA key size (16 bytes)
        else:
            key = bytes.fromhex(key)

        block_size = 8
        padded_message = message.encode().ljust((len(message) + block_size - 1) // block_size * block_size, b'\x00')

        ciphertext = bytearray()
        for i in range(0, len(padded_message), block_size):
            block = padded_message[i:i + block_size]
            encrypted_block = idea_encrypt_block(block, key)
            ciphertext.extend(encrypted_block)

        formatted_ciphertext = ' '.join(binascii.hexlify(ciphertext).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(ciphertext).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')
        
        return formatted_ciphertext, key_hex

    @staticmethod
    def rc4_encrypt(message, key=None):
        if key is None:
            key = get_random_bytes(16)  # RC4 key size
        else:
            key = bytes.fromhex(key)
        
        cipher = ARC4.new(key)
        encrypted_message = cipher.encrypt(message.encode())

        formatted_ciphertext = ' '.join(binascii.hexlify(encrypted_message).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(encrypted_message).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')
        
        return formatted_ciphertext, key_hex

    @staticmethod
    def twofish_encrypt(message, key=None):
        if key is None:
            key = get_random_bytes(16)  # Placeholder key size
        else:
            key = bytes.fromhex(key)

        cipher = AES.new(key, AES.MODE_ECB)  # Placeholder for Twofish
        padded_message = pad(message.encode(), AES.block_size)
        ciphertext = cipher.encrypt(padded_message)

        formatted_ciphertext = ' '.join(binascii.hexlify(ciphertext).decode('utf-8')[i:i+2] for i in range(0, len(binascii.hexlify(ciphertext).decode('utf-8')), 2))
        key_hex = binascii.hexlify(key).decode('utf-8')
        
        return formatted_ciphertext, key_hex
    
    def generate_ciphertext(self, paragraph):
        ciphertexts = []
        for algorithm, name in self.algorithms:
            ciphertext, key_hex = algorithm(paragraph)
            ciphertexts.append((ciphertext, name))
        return ciphertexts



import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft
from scipy.stats import entropy, norm, kstest
import re

class LinearCryptAnalyzer:
    def __init__(self, block_size=16, order=3):
        self.block_size = block_size
        self.order = order
    
    def split_into_blocks(self, ciphertext):
        blocks = []
        for i in range(0, len(ciphertext), self.block_size):
            block = ciphertext[i:i + self.block_size]
            if len(block) < self.block_size:
                block.extend([0] * (self.block_size - len(block)))
            blocks.append(block)
        return blocks

    def linear_cryptanalysis_blocks(self, blocks):
        num_blocks = len(blocks)
        linear_results = []
        block_indices = list(range(num_blocks))
        for combo in combinations(block_indices, self.order):
            xor_result = np.zeros(len(blocks[0]), dtype=np.int32)
            for idx in combo:
                xor_result ^= blocks[idx]
            linear_correlation = np.mean(xor_result == 0)  # Mean of zero results
            linear_results.append(linear_correlation)
        return np.array(linear_results)  # Return as numpy array for further processing

    def normalize_data(self, data):
        if len(data) == 0:
            print("No data to normalize.")
            return None
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        return normalized_data

    def statistical_analysis(self, data):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        mean = np.mean(data)
        var = np.var(data)
        unique, counts = np.unique(data, return_counts=True)
        frequencies = dict(zip(unique, counts))
        return mean, var, frequencies

    def entropy_analysis(self, data):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        _, counts = np.unique(data, return_counts=True)
        prob_dist = counts / len(data)
        ent = entropy(prob_dist, base=2)
        return ent

    def fourier_analysis(self, data):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        fft_result = fft(data)
        return np.abs(fft_result)

    def distribution_analysis(self, data):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        mu, std = norm.fit(data)
        _, p_value = kstest(data, 'norm', args=(mu, std))
        return mu, std, p_value

    def autocorrelation_analysis(self, data):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        n = len(data)
        mean = np.mean(data)
        var = np.var(data)
        autocorr = np.correlate(data - mean, data - mean, mode='full')
        autocorr = autocorr / (var * n)
        autocorr = autocorr[n-1:]
        return autocorr

    def heatmap_analysis(self, data):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        num_blocks = len(data) // self.block_size
        if num_blocks == 0:
            print("Insufficient data for heatmap analysis.")
            return None
        try:
            data_matrix = np.array(data).reshape(num_blocks, self.block_size)
        except ValueError as e:
            print(f"Error reshaping data for heatmap: {e}")
            return None
        heatmap_data = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            for j in range(num_blocks):
                heatmap_data[i, j] = np.sum(np.bitwise_xor(data_matrix[i], data_matrix[j]))
        return heatmap_data

    def preprocess_and_convert_ciphertext(self, ciphertext):
        """
        Takes the input ciphertext and block size, preprocesses the ciphertext 
        (converts to uppercase, removes spaces), checks if the format is correct 
        (only hexadecimal characters), and returns the unified cipher output as a list of 
        numeric values. If the ciphertext length doesn't match the block size, it adds padding 
        and notifies the user. The block size can be specified; default is 8.
        """
        cleaned_ciphertext = ciphertext.replace(' ', '').upper()
        if not re.fullmatch(r'[0-9A-F]+', cleaned_ciphertext):
            raise ValueError("Invalid format: Ciphertext contains non-hexadecimal characters.")
        block_length = self.block_size * 2  # Each byte is represented by 2 hex digits
        if len(cleaned_ciphertext) % block_length != 0:
            padding_length = block_length - (len(cleaned_ciphertext) % block_length)
            # print(f"Notice: Ciphertext length does not match the block size. Adding {padding_length // 2} bytes of padding.")
            cleaned_ciphertext += '00' * (padding_length // 2)
        cipher_numeric = [int(cleaned_ciphertext[i:i+2], 16) for i in range(0, len(cleaned_ciphertext), 2)]
        return cipher_numeric

    def analyze(self, ciphertext):
        # Preprocess and convert ciphertext
        cipher_numeric = self.preprocess_and_convert_ciphertext(ciphertext)

        # Split into blocks
        blocks = self.split_into_blocks(cipher_numeric)

        # Perform linear cryptanalysis
        linear_results = self.linear_cryptanalysis_blocks(blocks)

        # Normalize data
        normalized_data = self.normalize_data(linear_results)

        # Perform various analyses
        mean, var, frequencies = self.statistical_analysis(normalized_data)
        entropy_val = self.entropy_analysis(normalized_data)
        fourier_magnitudes = self.fourier_analysis(normalized_data)
        mu, std, p_value = self.distribution_analysis(normalized_data)
        autocorr = self.autocorrelation_analysis(normalized_data)
        # heatmap_data = self.heatmap_analysis(normalized_data)

        return {
            "Linear Cryptanalysis Results": linear_results,
            "Normalized Data": normalized_data,
            "Statistical Analysis": {"Mean": mean, "Variance": var, "Frequencies": frequencies},
            "Entropy": entropy_val,
            "Fourier Transform Magnitudes": fourier_magnitudes,
            "Distribution Analysis": {"Mean": mu, "Std": std, "p-value": p_value},
            "Autocorrelation": autocorr,
            # "Heatmap Data": heatmap_data
        }










def generate_random_text(length=100):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np

def handle_invalid_values(array):
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    return array

def compute_autocorrelation(data):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    if var == 0:
        return np.zeros(n)  # Return zero if variance is zero
    autocorr = np.correlate(data - mean, data - mean, mode='full')
    autocorr = autocorr / (var * n)
    return handle_invalid_values(autocorr)

def prepare_data(crypto_utils, analyzer, num_samples=100000, text_length=100):
    X = []
    y = []
    methods = [name for _, name in crypto_utils.algorithms]
    feature_length = 194  # Define the fixed length based on your feature extraction
    
    for _ in range(num_samples):
        text = generate_random_text(text_length)
        ciphertexts = crypto_utils.generate_ciphertext(text)
        for ciphertext, method in ciphertexts:
            features = analyzer.analyze(ciphertext)
            
            linear_results = features.get("Linear Cryptanalysis Results", np.zeros(10))
            normalized_data = features.get("Normalized Data", np.zeros(10))
            mean = features.get("Statistical Analysis", {}).get("Mean", 0)
            variance = features.get("Statistical Analysis", {}).get("Variance", 0)
            entropy = features.get("Entropy", 0)
            fourier_magnitudes = features.get("Fourier Transform Magnitudes", np.zeros(10))
            dist_mean = features.get("Distribution Analysis", {}).get("Mean", 0)
            dist_std = features.get("Distribution Analysis", {}).get("Std", 0)
            dist_pvalue = features.get("Distribution Analysis", {}).get("p-value", 0)
            autocorrelation = features.get("Autocorrelation", np.zeros(10))
            heatmap_data = features.get("Heatmap Data", np.zeros((3, 16))).flatten()
            
            feature_vector = np.concatenate([
                linear_results,
                normalized_data,
                [mean],
                [variance],
                [entropy],
                fourier_magnitudes,
                [dist_mean],
                [dist_std],
                [dist_pvalue],
                autocorrelation,
                heatmap_data
            ])
            
            # Ensure the feature vector is of fixed length
            if len(feature_vector) < feature_length:
                feature_vector = np.pad(feature_vector, (0, feature_length - len(feature_vector)))
            elif len(feature_vector) > feature_length:
                feature_vector = feature_vector[:feature_length]
            
            # Debugging print statement
            print(f"Feature vector shape: {feature_vector.shape}")
            
            X.append(feature_vector)
            y.append(method)
    
    X = np.array(X)
    y = np.array(y)
    return X, y





# Initialize CryptoUtils and LinearCryptAnalyzer
crypto_utils = CryptoUtils()
analyzer = LinearCryptAnalyzer(block_size=16, order=3)

# Prepare data
X, y = prepare_data(crypto_utils, analyzer)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)



import tensorflow as tf

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")



