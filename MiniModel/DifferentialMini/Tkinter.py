import tkinter as tk
from tkinter import ttk
from threading import Thread

import json
import numpy as np
import tensorflow as tf


import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import norm, kstest
from sklearn.preprocessing import StandardScaler
import re

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

# Load the model and label encoder
model = tf.keras.models.load_model('diffMiniv1-0.h5')
import joblib
label_encoder = joblib.load('labelEncoder_DiffMiniv1-0.pkl')

class CipherAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cipher Analyzer")

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Ciphertext entry
        tk.Label(self.root, text="Enter Ciphertext:").pack(pady=5)
        self.ciphertext_entry = tk.Entry(self.root, width=50)
        self.ciphertext_entry.pack(pady=5)

        # Analyze button
        self.analyze_button = tk.Button(self.root, text="Analyze", command=self.analyze_ciphertext)
        self.analyze_button.pack(pady=5)

        # Result text box
        self.result_text = tk.Text(self.root, height=15, width=80)
        self.result_text.pack(pady=5)

        # Loading animation
        self.loading_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.loading_label.pack(pady=5)

    def analyze_ciphertext(self):
        ciphertext = self.ciphertext_entry.get()
        if not ciphertext:
            tk.messagebox.showerror("Error", "Ciphertext cannot be empty!")
            return

        # Start a thread for analysis to keep UI responsive
        self.loading_label.config(text="Analyzing...")
        self.root.update_idletasks()
        analysis_thread = Thread(target=self.perform_analysis, args=(ciphertext,))
        analysis_thread.start()

    def perform_analysis(self, ciphertext):
        # Analyze ciphertext
        try:
            analyzer = CiphertextAnalyzer(ciphertext)
            result = analyzer.analyze()
            
            # For simplicity, we just display the statistical analysis and entropy
            display_result = json.dumps({
                'Statistical Analysis': result.get('Statistical Analysis', {}),
                'Entropy': result.get('Entropy', None)
            }, indent=4)

            # Run model prediction (example)
            features = np.array([self.extract_features(result)])
            prediction = model.predict(features)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
            
            display_result += f"\nPredicted Label: {predicted_label[0]}"

        except Exception as e:
            display_result = f"Error during analysis: {str(e)}"
        
        # Update the UI
        self.root.after(0, self.update_result, display_result)

    def extract_features(self, result):
        # Placeholder for feature extraction logic
        mean_diff = result.get('Statistical Analysis', {}).get('Mean', 0)
        variance_diff = result.get('Statistical Analysis', {}).get('Variance', 0)
        entropy_value = result.get('Entropy', 0)
        fourier_magnitudes = np.mean(result.get('Fourier Transform Magnitudes', [0]))
        return [mean_diff, variance_diff, entropy_value, fourier_magnitudes]

    def update_result(self, result):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)
        self.loading_label.config(text="Analysis Complete")

if __name__ == "__main__":
    root = tk.Tk()
    app = CipherAnalyzerApp(root)
    root.mainloop()
