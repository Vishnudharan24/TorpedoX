import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import norm, kstest
from sklearn.preprocessing import StandardScaler
import re



# Load the saved model
model = tf.keras.models.load_model('diffMiniv1.1.h5')

# Load the label encoder (assuming you saved it during training)
import joblib
label_encoder = joblib.load('labelEncoder_DiffMiniv1.1.pkl')




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





def preprocess_ciphertext(ciphertext, block_size=8):
    # Preprocess the ciphertext for analysis
    analyzer = CiphertextAnalyzer(ciphertext, block_size)
    analysis_result = analyzer.analyze()

    # Extract features for prediction
    if 'Statistical Analysis' in analysis_result:
        mean_diff = analysis_result['Statistical Analysis']['Mean']
        variance_diff = analysis_result['Statistical Analysis']['Variance']
        entropy_value = analysis_result['Entropy']
        fourier_magnitudes = np.mean(analysis_result['Fourier Transform Magnitudes'])
        # mu, std, p_value = analysis_result['Distribution Analysis']
        autocorr = np.mean(analysis_result['Autocorrelation'])
        heatmap_data = np.mean(analysis_result['Heatmap Data'])
        clustering_coefficients = np.mean(analysis_result['Cluster Centers'])
        
        features = np.array([mean_diff,variance_diff, entropy_value, fourier_magnitudes,  autocorr, heatmap_data, clustering_coefficients])
        return features.reshape(1, -1)  # Model expects a 2D array
    else:
        raise ValueError("Ciphertext analysis did not produce valid features.")

    
# def preprocess_ciphertext(ciphertext, block_size=8):
#     # Preprocess the ciphertext for analysis
#     analyzer = CiphertextAnalyzer(ciphertext, block_size)
#     analysis_result = analyzer.analyze()

#     # Extract features for prediction
#     features = []

#     # Statistical Analysis
#     if 'Statistical Analysis' in analysis_result:
#         mean_diff = analysis_result['Statistical Analysis']['Mean']
#         variance_diff = analysis_result['Statistical Analysis']['Variance']
#         features.extend([mean_diff, variance_diff])

#     # Entropy
#     if 'Entropy' in analysis_result:
#         entropy_value = analysis_result['Entropy']
#         features.append(entropy_value)

#     # Fourier Transform Magnitudes
#     if 'Fourier Transform Magnitudes' in analysis_result:
#         fourier_magnitudes = np.mean(analysis_result['Fourier Transform Magnitudes'])
#         features.append(fourier_magnitudes)

#     # Distribution Analysis
#     if 'Distribution Analysis' in analysis_result:
#         mu = analysis_result['Distribution Analysis']['Mean']
#         std = analysis_result['Distribution Analysis']['Std']
#         p_value = analysis_result['Distribution Analysis']['p-value']
#         features.extend([mu, std, p_value])

#     # Autocorrelation
#     if 'Autocorrelation' in analysis_result:
#         autocorrelation = analysis_result['Autocorrelation']
#         features.extend(np.mean(autocorrelation))  # You might need to adjust this depending on the autocorrelation shape

#     # Heatmap Data
#     if 'Heatmap Data' in analysis_result:
#         heatmap_data = analysis_result['Heatmap Data']
#         if heatmap_data is not None and len(heatmap_data) > 0:
#             heatmap_flat = heatmap_data.flatten()
#             features.extend(np.mean(heatmap_flat))  # Flattening and taking mean as example

#     # Optional: Add clustering features if clustering is performed
#     # Assuming `clustering_result` is a list of clustering features obtained elsewhere
#     if 'Clustering Results' in analysis_result:
#         clustering_features = analysis_result['Clustering Results']
#         features.extend(clustering_features)
    
#     return np.array(features).reshape(1, -1)  # Model expects a 2D array


def predict_algorithm(ciphertext):
    features = preprocess_ciphertext(ciphertext)
    # Predict the probabilities for each class
    prediction_probabilities = model.predict(features)
    # Get the indices of the classes
    class_indices = np.arange(prediction_probabilities.shape[1])
    # Map the probabilities to the labels
    probabilities = dict(zip(label_encoder.classes_, prediction_probabilities[0]))

    return probabilities

# Example usage
ciphertext = 'af dd c9 6c d5 4c 1e af 90 1d 7a b4 e8 35 14 ea 39 b0 96 29 b1 20 f4 6f 97 0e 10 e1 16 05 3f ac f4 c3 5c 11 a0 ba 3f e6 62 39 cc af e2 6f 96 91 19 08 b8 94 a9 f7 d3 a5 db 66 e5 a2 d0 52 3e 71 ec e1 d8 6c f9 92 34 d6 01 2c 24 51 cc 53 0e bc f4 18 9f 8e 2a d2 2d 93 49 30 16 a1 75 bd 9a 16 04 1f 5a 3c d2 de e5 41 05 13 6a f4 87 e8 70 99 3e 1c 35 c5 3b ce 81 e9 99 ae a3 31 43 2d 82 e4 01 44 9e 58 07 69 aa de e4 8c 0e cf 0e 0f 89 1f'
predicted_probabilities = predict_algorithm(ciphertext)
print("Prediction probabilities for each algorithm:")
for algo, prob in predicted_probabilities.items():
    print(f"{algo}: {prob:.4f}")

