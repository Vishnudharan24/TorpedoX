import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import string
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Crypto.Cipher import DES, AES, Blowfish
from Crypto.Random import get_random_bytes
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History

# Assuming `CiphertextAnalyzer` and `CryptoUtils` classes are already defined
from Crypto.Cipher import DES, DES3, AES, Blowfish, ARC4
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from Crypto.Cipher import ChaCha20_Poly1305, Salsa20
import binascii
import matplotlib.pyplot as plt
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
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os


from tqdm import tqdm  # Import tqdm for progress bars

class CiphertextAnalyzer:
    def __init__(self, ciphertext, block_size=8):
        self.ciphertext = ciphertext
        self.block_size = block_size
        self.cipher_numeric = self.preprocess_and_convert_ciphertext(ciphertext, block_size)
        self.differences = self.differential_cryptanalysis(self.cipher_numeric, block_size)
        self.result_normalized = self._normalize_result()
        
    def _normalize_result(self):
        if len(self.differences) > 0:
            scaler = StandardScaler()
            return scaler.fit_transform(self.differences.reshape(-1, 1)).flatten()
        return np.array([])

    def z_score_normalization(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        mean = np.mean(differences)
        std = np.std(differences)
        z_scores = (differences - mean) / std
        return z_scores

    def cross_correlation_analysis(self, differences1, differences2):
        if len(differences1) == 0 or len(differences2) == 0:
            print("No differences to analyze.")
            return None
        cross_corr = np.correlate(differences1, differences2, mode='full')
        return cross_corr

    def autocorrelation_analysis(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        n = len(differences)
        mean = np.mean(differences)
        var = np.var(differences)
        autocorr = np.correlate(differences - mean, differences - mean, mode='full')
        autocorr = autocorr / (var * n)
        autocorr = autocorr[n-1:]
        return autocorr

    def tsne_analysis(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        differences_reshaped = differences.reshape(-1, 1)
        if differences_reshaped.shape[0] < 2:
            print("Not enough data for t-SNE.")
            return None
        n_components_pca = min(differences_reshaped.shape[0], 10)
        pca = PCA(n_components=n_components_pca)
        pca_result = pca.fit_transform(differences_reshaped)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_result = tsne.fit_transform(pca_result)
        return tsne_result

    def distribution_analysis(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        mu, std = norm.fit(differences)
        _, p_value = kstest(differences, 'norm', args=(mu, std))
        return mu, std, p_value

    def shannon_entropy(self, data):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        _, counts = np.unique(data, return_counts=True)
        prob_dist = counts / len(data)
        entropy_value = -np.sum(prob_dist * np.log2(prob_dist + np.finfo(float).eps))
        return entropy_value

    def clustering_analysis(self, differences, num_clusters=3):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None, None
        differences_reshaped = differences.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(differences_reshaped)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        return cluster_centers, labels

    def correlation_analysis(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        correlation_matrix = np.corrcoef(differences)
        return correlation_matrix

    def statistical_analysis(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        mean_diff = np.mean(differences)
        var_diff = np.var(differences)
        unique, counts = np.unique(differences, return_counts=True)
        frequencies = dict(zip(unique, counts))
        return mean_diff, var_diff, frequencies

    def entropy_analysis(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        _, counts = np.unique(differences, return_counts=True)
        prob_dist = counts / len(differences)
        ent = entropy(prob_dist, base=2)
        return ent

    def fourier_analysis(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        diff_fft = fft(differences)
        return np.abs(diff_fft)

    def pca_analysis(self, differences):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        differences_reshaped = differences.reshape(-1, 1)
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(differences_reshaped)
        return pca_result

    def heatmap_analysis(self, differences, block_size):
        if len(differences) == 0:
            print("No differences to analyze.")
            return None
        num_blocks = len(differences) // block_size
        difference_matrix = np.array(differences).reshape(num_blocks, block_size)
        heatmap_data = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            for j in range(num_blocks):
                heatmap_data[i, j] = np.sum(np.bitwise_xor(difference_matrix[i].astype(int), difference_matrix[j].astype(int)))
        return heatmap_data

    def differential_cryptanalysis(self, ciphertext, block_size):
        blocks = []
        differences = []
        for i in range(0, len(ciphertext), block_size):
            block = ciphertext[i:i + block_size]
            if len(block) < block_size:
                block.extend([0] * (block_size - len(block)))
            blocks.append(block)
        if len(blocks) < 2:
            return np.array([])
        for i in range(len(blocks) - 1):
            diff = np.bitwise_xor(blocks[i], blocks[i + 1])
            differences.append(diff)
        if differences:
            return np.hstack(differences)
        else:
            print("returning empty array")
            return np.array([])

    def preprocess_and_convert_ciphertext(self, ciphertext, block_size=16):
        cleaned_ciphertext = ciphertext.replace(' ', '').upper()
        if not re.fullmatch(r'[0-9A-F]+', cleaned_ciphertext):
            raise ValueError("Invalid format: Ciphertext contains non-hexadecimal characters.")
        block_length = block_size * 2
        if len(cleaned_ciphertext) % block_length != 0:
            padding_length = block_length - (len(cleaned_ciphertext) % block_length)
            # print(f"Notice: Ciphertext length does not match the block size. Adding {padding_length // 2} bytes of padding.")
            cleaned_ciphertext += '00' * (padding_length // 2)
        cipher_numeric = [int(cleaned_ciphertext[i:i+2], 16) for i in range(0, len(cleaned_ciphertext), 2)]
        return cipher_numeric

    def analyze(self):
        results = {}
        if len(self.result_normalized) > 0:
            mean_diff, var_diff, frequencies = self.statistical_analysis(self.result_normalized)
            correlation_matrix = self.correlation_analysis(self.result_normalized)
            entropy_value = self.entropy_analysis(self.result_normalized)
            fourier_values = self.fourier_analysis(self.result_normalized)
            heatmap_data = self.heatmap_analysis(self.result_normalized, self.block_size)
            cluster_centers, labels = self.clustering_analysis(self.result_normalized)
            shannon_entropy_value = self.shannon_entropy(self.result_normalized)
            mu, std, p_value = self.distribution_analysis(self.result_normalized)
            autocorr = self.autocorrelation_analysis(self.result_normalized)

            results['Statistical Analysis'] = {
                'Mean': mean_diff,
                'Variance': var_diff,
                'Frequencies': frequencies
            }
            results['Correlation Matrix'] = correlation_matrix
            results['Entropy'] = entropy_value
            results['Shannon Entropy'] = shannon_entropy_value
            results['Fourier Transform Magnitudes'] = fourier_values
            results['Heatmap Data'] = heatmap_data
            results['Cluster Centers'] = cluster_centers
            results['Labels'] = labels
            results['Distribution Analysis'] = {
                'Mean': mu,
                'Std': std,
                'p-value': p_value
            }
            results['Autocorrelation'] = autocorr
        else:
            results['Error'] = "No data available for further analysis."

        return results
    
    
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



def generate_random_message(length=1000):
    """Generate a random string of specified length."""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))



def analyze_sample(ciphertext, encrypt_func, label):
    analyzer = CiphertextAnalyzer(ciphertext)
    analysis_result = analyzer.analyze()
    
    if 'Statistical Analysis' in analysis_result:
        mean_diff = analysis_result['Statistical Analysis']['Mean']
        variance_diff = analysis_result['Statistical Analysis']['Variance']
        entropy_value = analysis_result['Entropy']
        fourier_magnitudes = np.mean(analysis_result['Fourier Transform Magnitudes'])
        
        
        
        #             results['Statistical Analysis'] = {
        #         'Mean': mean_diff,
        #         'Variance': var_diff,
        #         'Frequencies': frequencies
        #     }
        #     results['Correlation Matrix'] = correlation_matrix
        #     results['Entropy'] = entropy_value
        #     results['Shannon Entropy'] = shannon_entropy_value
        #     results['Fourier Transform Magnitudes'] = fourier_values
        #     results['Heatmap Data'] = heatmap_data
        #     results['Cluster Centers'] = cluster_centers
        #     results['Labels'] = labels
        #     results['Distribution Analysis'] = {
        #         'Mean': mu,
        #         'Std': std,
        #         'p-value': p_value
        #     }
        #     results['Autocorrelation'] = autocorr
        # else:
        #     results['Error'] = "No data available for further analysis."
        
        return [mean_diff, variance_diff, entropy_value, fourier_magnitudes], label
    return None, None

def generate_data(num_samples, batch_size=100):
    crypto_utils = CryptoUtils()
    features = []
    labels = []

    # Use ProcessPoolExecutor with a limit on the number of workers
    with ProcessPoolExecutor(max_workers=7) as executor:  # Adjust max_workers as needed
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Generating Samples in Batches"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_futures = []

            for _ in range(batch_start, batch_end):
                for encrypt_func, label in crypto_utils.algorithms:
                    message = generate_random_message()
                    ciphertext, _ = encrypt_func(message)
                    batch_futures.append(executor.submit(analyze_sample, ciphertext, encrypt_func, label))
            
            # Collect results from the batch with a progress bar
            for future in tqdm(as_completed(batch_futures), total=len(batch_futures), desc="Processing Batch"):
                result, label = future.result()
                if result is not None:
                    features.append(result)
                    labels.append(label)

    return np.array(features), np.array(labels)

def train_and_save_model(features, labels):
    # Encoding labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.3, random_state=42)
    
    # Initialize the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(np.unique(labels_encoded)), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model and save the history
    history = model.fit(X_train, y_train,
                        epochs=5,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1)

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save the model and label encoder
    model_filename = 'DiffMini_v1.0s2(acclerated).h5'
    encoder_filename = 'DiffMini_v1.0s2(acclerated).pkl'
    
    model.save(model_filename)
    joblib.dump(label_encoder, encoder_filename)
    
    print(f'Model saved as {model_filename}')
    print(f'Label encoder saved as {encoder_filename}')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    num_samples = 5000  # Define the number of samples to generate
    features, labels = generate_data(num_samples)
    train_and_save_model(features, labels)
   