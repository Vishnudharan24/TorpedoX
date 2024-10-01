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
import string 

from Crypto.Cipher import DES, DES3, AES, Blowfish, ARC4
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from Crypto.Cipher import ChaCha20_Poly1305, Salsa20
import binascii

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from Crypto.Cipher import AES, DES, Blowfish
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
import binascii

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# from gutenberg.acquire import load_etext
# from gutenberg.cleanup import strip_headers



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
















class CiphertextAnalyzer:#Differential 
    def __init__(self, ciphertext, block_size=8):
        self.ciphertext = ciphertext
        self.block_size = block_size
        #self.cipher_numeric = self.preprocess_and_convert_ciphertext(ciphertext, block_size)
        #self.differences = self.differential_cryptanalysis(self.cipher_numeric, block_size)
        #self.result_normalized = self._normalize_result()
        
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

    def differential_cryptanalysis(self, ciphertext, block_size=16):
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
            return np.array([])

    def preprocess_and_convert_ciphertext(self, ciphertext, block_size=8):
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

    def analyze(self, ciphertext):
        # Step 1: Preprocess and convert the ciphertext
        preprocessed_data = self.preprocess_and_convert_ciphertext(ciphertext)
        
        # Step 2: Perform differential cryptanalysis
        differential_data = self.differential_cryptanalysis(preprocessed_data)
        
        # Step 3: Perform analysis on the result of differential cryptanalysis
        results = {}
        
        if len(differential_data) > 0:
            mean_diff, var_diff, frequencies = self.statistical_analysis(differential_data)
            correlation_matrix = self.correlation_analysis(differential_data)
            entropy_value = self.entropy_analysis(differential_data)
            fourier_values = self.fourier_analysis(differential_data)
            heatmap_data = self.heatmap_analysis(differential_data, self.block_size)
            cluster_centers, labels = self.clustering_analysis(differential_data)
            shannon_entropy_value = self.shannon_entropy(differential_data)
            mu, std, p_value = self.distribution_analysis(differential_data)
            autocorr = self.autocorrelation_analysis(differential_data)

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

    
    
class HigherOrderAnalyzer:
    def __init__(self, ciphertext, block_size=16):
        self.ciphertext = ciphertext
        self.block_size = block_size
        # self.cipher_numeric = self.preprocess_and_convert_ciphertext(ciphertext, block_size)
        # self.blocks = self.split_into_blocks(self.cipher_numeric, block_size)
        self.correlations = None
        self.correlations_normalized = None
    
    def preprocess_and_convert_ciphertext(self, ciphertext, block_size=16):
        print(ciphertext)
        cleaned_ciphertext = ciphertext.replace(' ', '').upper()
        if not re.fullmatch(r'[0-9A-F]+', cleaned_ciphertext):
            print("cleaned_cipher_text",cleaned_ciphertext)
            raise ValueError("Invalid format: Ciphertext contains non-hexadecimal characters.")
        block_length = block_size * 2
        if len(cleaned_ciphertext) % block_length != 0:
            padding_length = block_length - (len(cleaned_ciphertext) % block_length)
            print(f"Notice: Ciphertext length does not match the block size. Adding {padding_length // 2} bytes of padding.")
            cleaned_ciphertext += '00' * (padding_length // 2)
        cipher_numeric = [int(cleaned_ciphertext[i:i+2], 16) for i in range(0, len(cleaned_ciphertext), 2)]
        return cipher_numeric

    def split_into_blocks(self, ciphertext, block_size=16):
        blocks = []
        for i in range(0, len(ciphertext), block_size):
            block = ciphertext[i:i + block_size]
            if len(block) < block_size:
                block.extend([0] * (block_size - len(block)))
            blocks.append(block)
        return blocks

    def higher_order_linear_cryptanalysis_blocks(self,blocks, order=3):
        self.blocks = blocks 
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

    def heatmap_analysis(self, data, block_size):
        if len(data) == 0:
            print("No data to analyze.")
            return None
        num_blocks = len(data) // block_size
        if num_blocks == 0:
            print("Insufficient data for heatmap analysis.")
            return None
        try:
            # Convert data to a numpy array of uint8 for bitwise operations
            data_matrix = np.array(data, dtype=np.uint8).reshape(num_blocks, block_size)
        except ValueError as e:
            print(f"Error reshaping data for heatmap: {e}")
            return None
        heatmap_data = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            for j in range(num_blocks):
                # Compute bitwise XOR and count the number of differing bits
                heatmap_data[i, j] = np.sum(np.bitwise_xor(data_matrix[i], data_matrix[j]))
        return heatmap_data
    
    def analyze(self, ciphertext):
        # Preprocess and convert the ciphertext
        numeric_cipher = self.preprocess_and_convert_ciphertext(ciphertext)
        
        # Split the numeric cipher into blocks
        blocks = self.split_into_blocks(numeric_cipher)
        
        # Perform higher-order linear cryptanalysis on the blocks and store the result
        cryptanalysis_result = self.higher_order_linear_cryptanalysis_blocks(blocks)
        
        # Normalize the cryptanalysis result
        self.correlations_normalized = self.normalize_data(cryptanalysis_result)
        
        # Perform remaining analyses
        mean_corr, var_corr, freq_corr = self.statistical_analysis(self.correlations_normalized)
        entropy_corr = self.entropy_analysis(self.correlations_normalized)
        fourier_corr = self.fourier_analysis(self.correlations_normalized)
        mu, std, p_value = self.distribution_analysis(self.correlations_normalized)
        autocorr = self.autocorrelation_analysis(self.correlations_normalized)
        heatmap_data = self.heatmap_analysis(self.correlations_normalized, self.block_size)
        
        # Return results
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
            "Autocorrelation": autocorr,
            "Heatmap Data": heatmap_data
        }

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
            print(f"Notice: Ciphertext length does not match the block size. Adding {padding_length // 2} bytes of padding.")
            cleaned_ciphertext += '00' * (padding_length // 2)
        cipher_numeric = [int(cleaned_ciphertext[i:i+2], 16) for i in range(0, len(cleaned_ciphertext), 2)]
        return cipher_numeric

    def analyze(self, ciphertext):
        # Preprocess and convert ciphertext
        # print("Linear analysis :",ciphertext)
        cipher_numeric = self.preprocess_and_convert_ciphertext(ciphertext)

        # Split into blocks
        blocks = self.split_into_blocks(cipher_numeric)
        print("Blocks",blocks)
        # Perform linear cryptanalysis
        linear_results = self.linear_cryptanalysis_blocks(blocks)
        print("LInear results",linear_results)
        # Normalize data
        normalized_data = self.normalize_data(linear_results)
        print("Normalized Data",normalized_data)

        # Perform various analyses
        mean, var, frequencies = self.statistical_analysis(normalized_data)
        entropy_val = self.entropy_analysis(normalized_data)
        fourier_magnitudes = self.fourier_analysis(normalized_data)
        mu, std, p_value = self.distribution_analysis(normalized_data)
        autocorr = self.autocorrelation_analysis(normalized_data)
        heatmap_data = self.heatmap_analysis(normalized_data)

        return {
            "Linear Cryptanalysis Results": linear_results,
            "Normalized Data": normalized_data,
            "Statistical Analysis": {"Mean": mean, "Variance": var, "Frequencies": frequencies},
            "Entropy": entropy_val,
            "Fourier Transform Magnitudes": fourier_magnitudes,
            "Distribution Analysis": {"Mean": mu, "Std": std, "p-value": p_value},
            "Autocorrelation": autocorr,
            "Heatmap Data": heatmap_data
        }
        
        
# ciphertext = 'da 29 01 2c c2 c4 f4 65 24 a2 7f ef f7 bd c0 a2 40 a0 e3 7e 8e 70 b4 cc d5 91 c6 f9 4b c0 03 eb 8f e8 0d a7 90 52 11 16 40 43 98 50 ee af 8f 11 a2 21 4a 34 e0 84 21 e4 34 97 07 b1 99 7c f0 16 7a db 96 dd 3b ae 41 c6 1d 0d 81 ed 3a 2e 2e 17 11 a9 23 53 3d 3f b5 60 51 21 d2 60 cf 4c 35 3b 8b 79 1e 32 85 db f6 8e 77 0f 3a 08 cb c0 54 40 4d 5d 20 6c d6 49 8c 98 e3 5d 0c f4 b6 44 1e 2d af 03 af d3 40 b5 cd 19 2d 79 ff bd 3c 69 59 73 d9 f7 50 87 ea b4 45 52 7b d8 4f 6c 9e 7c bc bf 9d fa bb f4 36 b6 4e c4 a8 b2 80 20 0e 9b 43 16 35 60 4c 2e 96 5e 1c ba 44 c6 21 0b 4c 4d 77 bd 20 c0 27 b0 dd 6c 51 f0 0d 8f 28 52 57 53 3c 18 9f 2c 85 6d dd 65 cc 74 36 4e 50 d9 44 4e 5f 4e 62 56 58 b9 eb a1 dc aa 30 5d 9f e6 96 92 11 d3 58 a8 21 69 78 f5 84 bc 82 b2 8b ab 4c ce e4 14 02 0c 8b 54 f4 43 4c 4e 44 f5 b5 71 0a 3b 0f c8 0f 72 77 45 b9 70 4d 60 5e 9b 57 8c 22 1d 17 5a 3e d0 4e 2d a4 d1 df 4d 1d 8b 8c 06 26 a1 7d 35 42 4d 66 90 91 c1 eb 98 18 a2 5f c1 c5 a2 cb 51 bd 78 6d 37 55 d7 de 15 ba 2f 5e d7 73 51 d8 81 81 3a fa ea c5 04 20 22 d1 f4 87 52 d6 6f 0c 6b a3 fd f3 a9 0b c4 48 84 47 b2 d3 cd 38 5d fd 12 ea d4 fd 24 0e c3 99 b9 d1 f7 86 19 a3 53 8e cc 45 87 bf 20 83 aa d1 41 4f 93 82 0f a8 c6 69 f6'
# block_size = 8
# order = 2
# analyzer = LinearCryptAnalyzer(block_size=block_size, order=order)
# results = analyzer.analyze(ciphertext)
# analyzer = HigherOrderAnalyzer(ciphertext, block_size=8)
# results = analyzer.analyze()
# analyzer = CiphertextAnalyzer(ciphertext, block_size=8)
# result = analyzer.analyze()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import random
import numpy as np
import nltk
from nltk.corpus import gutenberg
from sklearn.preprocessing import StandardScaler

# Ensure NLTK resources are downloaded
# nltk.download('gutenberg')

class CustomParagraphDataset(Dataset):
    def __init__(self, num_paragraphs, paragraph_size):
        self.num_paragraphs = num_paragraphs
        self.paragraph_size = paragraph_size
        self.data = self._generate_paragraphs()

    def _generate_paragraphs(self):
        paragraphs = []
        for _ in range(self.num_paragraphs):
            # Generate a random paragraph with random words
            paragraph = ' '.join(
                ''.join(random.choices(string.ascii_letters, k=random.randint(3, 10)))
                for _ in range(self.paragraph_size // 10)
            )
            paragraphs.append(paragraph)
        return paragraphs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paragraph = self.data[idx]
        return paragraph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_size=(1, 128, 128), num_classes=10):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Output: (32, H, W)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, H/2, W/2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: (64, H/2, W/2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, H/4, W/4)
        
        # Compute the size of the feature map after convolution and pooling
        self._compute_output_size()
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def _compute_output_size(self):
        # Compute feature size after convolution and pooling layers
        x = torch.randn(1, *self.input_size)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        self.feature_size = torch.prod(torch.tensor(x.size()[1:])).item()  # Flatten size after conv layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, self.feature_size)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Define other classes here: HigherOrderAnalyzer, CiphertextAnalyzer, LinearCryptAnalyzer, CryptoUtils, CustomParagraphDataset

class CNNDecisionFusion:
    def __init__(self, num_classes, paragraph_size, num_paragraphs):
        self.num_classes = num_classes
        self.paragraph_size = paragraph_size
        self.num_paragraphs = num_paragraphs
        
        # Initialize analyzers and model
        self.higher_order_analyzer = HigherOrderAnalyzer(ciphertext='AB', block_size=8)
        self.ciphertext_analyzer = CiphertextAnalyzer(ciphertext='AB')
        self.linear_analyzer = LinearCryptAnalyzer()
        self.model = SimpleCNN(input_size=(1, 32, 112), num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Initialize CryptoUtils and algorithm-to-label mapping
        self.crypto_utils = CryptoUtils()
        self.algorithm_names = [
            'aes_encrypt',
            'des_encrypt',
            'blowfish_encrypt',
            'chacha20_poly1305_encrypt',
            'twofish_encrypt',
            'idea_encrypt',
            'rc4_encrypt',
            'salsa20_encrypt',
            'triple_des_encrypt'
        ]
        self.algorithm_to_label = {name: i for i, name in enumerate(self.algorithm_names)}

    def _preprocess_ciphertext(self, ciphertext):
        numeric_data = [int(c, 16) for c in ciphertext.split()]
        tensor_data = torch.tensor(numeric_data).float().unsqueeze(0).unsqueeze(0)
        return tensor_data

    def _combine_analysis_data(self, cnn_output, linear_results, higher_order_results, analysis_results):
        combined_data = torch.cat((cnn_output, linear_results, higher_order_results, analysis_results), dim=1)
        return combined_data

    def forward(self, input_tensor):
        cnn_output = self.model(input_tensor)
        return cnn_output

    def train_model(self, num_epochs=10, batch_size=16):
        dataset = CustomParagraphDataset(num_paragraphs=self.num_paragraphs, paragraph_size=self.paragraph_size)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for paragraphs in train_loader:
                for paragraph in paragraphs:
                    ciphertexts = self.crypto_utils.generate_ciphertext(paragraph)
                    
                    for ciphertext, algo_name in ciphertexts:
                        input_tensor = self._preprocess_ciphertext(ciphertext)
                        
                        linear_results = self.linear_analyzer.analyze(ciphertext)
                        higher_order_results = self.higher_order_analyzer.analyze(ciphertext)
                        analysis_results = self.ciphertext_analyzer.analyze(ciphertext)
                        
                        cnn_output = self.forward(input_tensor)
                        
                        combined_data = self._combine_analysis_data(cnn_output, linear_results, higher_order_results, analysis_results)
                        
                        label = torch.tensor(self.algorithm_to_label[algo_name]).unsqueeze(0)
                        
                        loss = self.criterion(combined_data, label)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}')



# Initialize and train the model
model = CNNDecisionFusion(num_classes=10, paragraph_size=128, num_paragraphs=10)
model.train_model(num_epochs=10, batch_size=16)
