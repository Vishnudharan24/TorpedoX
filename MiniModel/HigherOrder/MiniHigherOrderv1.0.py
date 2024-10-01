import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Crypto.Cipher import DES, AES, Blowfish
from Crypto.Random import get_random_bytes

# Assuming `CiphertextAnalyzer` and `CryptoUtils` classes are already defined
from Crypto.Cipher import DES, DES3, AES, Blowfish, ARC4
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from Crypto.Cipher import ChaCha20_Poly1305, Salsa20
import binascii

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import norm, kstest
from sklearn.preprocessing import StandardScaler
import re




import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft
from scipy.stats import entropy, norm, kstest
import re

from tqdm import tqdm

class HigherOrderAnalyzer:
    def __init__(self, ciphertext, block_size=16):
        self.ciphertext = ciphertext
        self.block_size = block_size
        self.cipher_numeric = self.preprocess_and_convert_ciphertext(ciphertext, block_size)
        self.blocks = self.split_into_blocks(self.cipher_numeric, block_size)
        self.correlations = None
        self.correlations_normalized = None
    
    def preprocess_and_convert_ciphertext(self, ciphertext, block_size):
        cleaned_ciphertext = ciphertext.replace(' ', '').upper()
        if not re.fullmatch(r'[0-9A-F]+', cleaned_ciphertext):
            raise ValueError("Invalid format: Ciphertext contains non-hexadecimal characters.")
        block_length = block_size * 2
        if len(cleaned_ciphertext) % block_length != 0:
            padding_length = block_length - (len(cleaned_ciphertext) % block_length)
            print(f"Notice: Ciphertext length does not match the block size. Adding {padding_length // 2} bytes of padding.")
            cleaned_ciphertext += '00' * (padding_length // 2)
        cipher_numeric = [int(cleaned_ciphertext[i:i+2], 16) for i in range(0, len(cleaned_ciphertext), 2)]
        return cipher_numeric

    def split_into_blocks(self, ciphertext, block_size):
        blocks = []
        for i in range(0, len(ciphertext), block_size):
            block = ciphertext[i:i + block_size]
            if len(block) < block_size:
                block.extend([0] * (block_size - len(block)))
            blocks.append(block)
        return blocks

    def higher_order_linear_cryptanalysis_blocks(self, order=3):
        num_blocks = len(self.blocks)
        correlations = []
        block_indices = list(range(num_blocks))
        for combo in combinations(block_indices, order):
            xor_result = np.zeros(len(self.blocks[0]), dtype=np.int32)
            for idx in combo:
                xor_result ^= self.blocks[idx]
            correlation = np.mean(xor_result == 0)
            correlations.append(correlation)
        self.correlations = np.array(correlations)
        return self.correlations

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


    def compute_autocorrelation(self,data):
        n = len(data)
        mean = np.mean(data)
        var = np.var(data)

        if var == 0:
            print("Variance is zero, which may lead to invalid results.")
            return np.zeros(n)  # or handle it appropriately

        autocorr = np.correlate(data - mean, data - mean, mode='full')
        autocorr = autocorr / (var * n)

        return autocorr


    def heatmap_analysis(self, data, block_size=2):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        num_blocks = len(data) // block_size
        if num_blocks == 0:
            print("Insufficient data for heatmap analysis.")
            return None
        try:
            data_matrix = np.array(data, dtype=np.int32).reshape(num_blocks, block_size)  # Ensure data is int32
        except ValueError as e:
            print(f"Error reshaping data for heatmap: {e}")
            return None
        heatmap_data = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            for j in range(num_blocks):
                heatmap_data[i, j] = np.sum(np.bitwise_xor(data_matrix[i], data_matrix[j]))
        return heatmap_data


    def analyze(self):
        if not hasattr(self, 'correlations') or self.correlations is None:
            self.higher_order_linear_cryptanalysis_blocks(order=2)  # Perform higher-order linear cryptanalysis if correlations is not set
        self.correlations_normalized = self.normalize_data(self.correlations)  # Normalize the correlations

        mean_corr, var_corr, freq_corr = self.statistical_analysis(self.correlations_normalized)
        entropy_corr = self.entropy_analysis(self.correlations_normalized)
        fourier_corr = self.fourier_analysis(self.correlations_normalized)
        mu, std, p_value = self.distribution_analysis(self.correlations_normalized)
        # autocorr = self.compute_autocorrelation(self.correlations_normalized)
        
        
        # heatmap_data = self.heatmap_analysis(self.correlations_normalized, block_size=3)
        # if heatmap_data is None:
        #     heatmap_data = np.zeros((len(self.correlations_normalized), len(self.correlations_normalized)))

        return {
            "Higher-Order Linear Cryptanalysis Result (Normalized)": self.correlations_normalized,
            "Statistical Analysis": {
                "Mean": mean_corr,
                "Variance": var_corr,
                "Frequencies": freq_corr
            },
            "Entropy": entropy_corr,
            "Fourier Transform Magnitudes": fourier_corr,
            "Distribution Analysis": {
                "Mean": mu,
                "Std": std,
                "p-value": p_value
            },
            # "Autocorrelation": autocorr,
            # "Heatmap Data": heatmap_data
        }





class CryptoUtils:
    
    def __init__(self):
        self.algorithms = [
            # (self.aes_encrypt, 'AES'),
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

# Step 1: Generate Data
import random
import string


def generate_random_message(length=10000):
    """Generate a random string of specified length."""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def generate_data(num_samples=100):
    crypto_utils = CryptoUtils()
    features = []
    labels = []

    # Use tqdm to wrap the range iterator to display the progress bar
    for _ in tqdm(range(num_samples), desc="Generating Samples"):
        for encrypt_func, label in crypto_utils.algorithms:
            message = generate_random_message()  # Generate a random message
            ciphertext, _ = encrypt_func(message)
            analyzer = HigherOrderAnalyzer(ciphertext)
            analysis_result = analyzer.analyze()

            # Collect the features (e.g., statistical analysis, entropy, etc.)
            if 'Statistical Analysis' in analysis_result:
                mean_diff = analysis_result['Statistical Analysis']['Mean']
                variance_diff = analysis_result['Statistical Analysis']['Variance']
                entropy_value = analysis_result['Entropy']
                fourier_magnitudes = np.mean(analysis_result['Fourier Transform Magnitudes'])
                
                features.append([mean_diff, variance_diff, entropy_value, fourier_magnitudes])
                labels.append(label)
                
        #                 return {
        #     "Higher-Order Linear Cryptanalysis Result (Normalized)": self.correlations_normalized,
        #     "Statistical Analysis": {
        #         "Mean": mean_corr,
        #         "Variance": var_corr,
        #         "Frequencies": freq_corr
        #     },
        #     "Entropy": entropy_corr,
        #     "Fourier Transform Magnitudes": fourier_corr,
        #     "Distribution Analysis": {
        #         "Mean": mu,
        #         "Std": std,
        #         "p-value": p_value
        #     },
        #     # "Autocorrelation": autocorr,
        #     # "Heatmap Data": heatmap_data
        # }

    return np.array(features), np.array(labels)




# Step 2: Prepare the Data
features, labels = generate_data(num_samples=100)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Step 3: Build the Neural Network
# Step 3: Build the Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Accuracy: {accuracy:.4f}')

# Save the model if needed
model.save('diffMiniv1.1.h5')

import joblib
joblib.dump(label_encoder, 'labelEncoder_DiffMiniv1.1.pkl')


# Plot training history if needed
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
