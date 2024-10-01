import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Crypto.Cipher import DES, DES3, AES, Blowfish, ARC4
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from Crypto.Cipher import ChaCha20_Poly1305, Salsa20
import binascii
import re
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import norm, kstest
from sklearn.preprocessing import StandardScaler

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
            return np.array([])

    def preprocess_and_convert_ciphertext(self, ciphertext, block_size=16):
        cleaned_ciphertext = ciphertext.replace(' ', '').upper()
        if not re.fullmatch(r'[0-9A-F]+', cleaned_ciphertext):
            raise ValueError("Invalid format: Ciphertext contains non-hexadecimal characters.")
        block_length = block_size * 2
        if len(cleaned_ciphertext) % block_length != 0:
            padding_length = block_length - (len(cleaned_ciphertext) % block_length)
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
    @staticmethod
    def generate_ciphertext(method, plaintext, key=None):
        key = key or get_random_bytes(32)  # Default key size to 32 bytes
        if method == 'DES':
            cipher = DES.new(key[:8], DES.MODE_ECB)
        elif method == 'TripleDES':
            cipher = DES3.new(key[:24], DES3.MODE_ECB)
        elif method == 'AES':
            cipher = AES.new(key[:16], AES.MODE_ECB)
        elif method == 'Blowfish':
            cipher = Blowfish.new(key[:16], Blowfish.MODE_ECB)
        elif method == 'ChaCha20':
            key = key[:32]  # Ensure key is 32 bytes
            cipher = ChaCha20_Poly1305.new(key=key)
        elif method == 'Salsa20':
            key = key[:16]  # Salsa20 requires 16 bytes
            cipher = Salsa20.new(key=key)
        elif method == 'ARC4':
            cipher = ARC4.new(key)
        else:
            raise ValueError(f"Unsupported cipher method: {method}")

        ciphertext = cipher.encrypt(pad(plaintext.encode(), 16))
        return binascii.hexlify(ciphertext).decode('utf-8'), binascii.hexlify(key).decode('utf-8')


class CNNDecisionFusion:
    def __init__(self, analysis_data):
        self.analysis_data = analysis_data

    def build_model(self):
        inputs = tf.keras.Input(shape=(len(self.analysis_data[0]),))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, X_test, y_test, epochs=10):
        model = self.build_model()
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
        return model

    def evaluate_model(self, model, X_test, y_test):
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, accuracy

# Sample usage
plaintext = "This is a secret message"
cipher_methods = ['DES', 'TripleDES', 'AES', 'Blowfish', 'ChaCha20', 'Salsa20', 'ARC4']
ciphertexts = []
keys = []

for method in cipher_methods:
    ciphertext, key = CryptoUtils.generate_ciphertext(method, plaintext)
    ciphertexts.append(ciphertext)
    keys.append(key)

analyzer = CiphertextAnalyzer(ciphertexts[0])
analysis_results = analyzer.analyze()
print(analysis_results)

# Assuming analysis_data and labels are prepared
X_train, X_test, y_train, y_test = train_test_split(np.array(analysis_results['Fourier Transform Magnitudes']), np.array([0]*len(ciphertexts)), test_size=0.2)
cnn_fusion = CNNDecisionFusion(analysis_results)
model = cnn_fusion.train_model(X_train, y_train, X_test, y_test)
loss, accuracy = cnn_fusion.evaluate_model(model, X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")


