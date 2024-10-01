import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.fft import fft
from scipy.stats import skew, kurtosis
import re

# Load the model and label encoder
model = tf.keras.models.load_model('cipher_algorithm_classifier.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Define CiphertextAnalyzer class
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
            mean_diff = np.mean(self.result_normalized)
            var_diff = np.var(self.result_normalized)
            entropy_value = self.shannon_entropy(self.result_normalized)
            fourier_magnitudes = np.mean(self.fourier_analysis(self.result_normalized))
            skewness_value = skew(self.result_normalized)
            kurtosis_value = kurtosis(self.result_normalized)

            results['Mean'] = mean_diff
            results['Variance'] = var_diff
            results['Entropy'] = entropy_value
            results['Fourier Magnitudes'] = fourier_magnitudes
            results['Skewness'] = skewness_value
            results['Kurtosis'] = kurtosis_value
        else:
            results['Error'] = "No data available for further analysis."

        return results

    def shannon_entropy(self, data):
        if len(data) == 0:
            return None
        _, counts = np.unique(data, return_counts=True)
        prob_dist = counts / len(data)
        entropy_value = -np.sum(prob_dist * np.log2(prob_dist + np.finfo(float).eps))
        return entropy_value

    def fourier_analysis(self, differences):
        if len(differences) == 0:
            return None
        diff_fft = fft(differences)
        return np.abs(diff_fft)

# Prepare PCA transformation
class PCAHandler:
    def __init__(self, n_components=5):
        self.pca = PCA(n_components=n_components)
    
    def fit_transform(self, features):
        if features.shape[0] < self.pca.n_components:
            # Not enough samples to apply PCA with the requested number of components
            return features
        return self.pca.fit_transform(features)

pca_handler = PCAHandler(n_components=5)

# Function to predict the encryption algorithm and return probabilities
def predict_algorithm(ciphertext):
    analyzer = CiphertextAnalyzer(ciphertext)
    analysis_result = analyzer.analyze()

    if 'Mean' in analysis_result:
        mean_diff = analysis_result['Mean']
        variance_diff = analysis_result['Variance']
        entropy_value = analysis_result['Entropy']
        fourier_magnitudes = analysis_result['Fourier Magnitudes']
        skewness_value = analysis_result['Skewness']
        kurtosis_value = analysis_result['Kurtosis']
        
        # Ensure the number of features matches the model's input
        features = np.array([[mean_diff, variance_diff, entropy_value, fourier_magnitudes, skewness_value]])
        
        # Apply PCA transformation
        features_pca = pca_handler.fit_transform(features)

        # Predict using the loaded model
        prediction = model.predict(features_pca)
        
        # Get the probabilities for each class
        probabilities = prediction[0]
        class_labels = label_encoder.classes_

        # Format the output
        probabilities_output = {class_labels[i]: probabilities[i] for i in range(len(class_labels))}
        return probabilities_output
    else:
        return {"Error": "No data available for further analysis."}

# Example usage
if __name__ == "__main__":
    ciphertext_input = input("Enter the ciphertext (hexadecimal format): ")
    probabilities = predict_algorithm(ciphertext_input)
    
    if "Error" in probabilities:
        print(probabilities["Error"])
    else:
        print("Class probabilities:")
        for label, prob in probabilities.items():
            print(f"{label}: {prob:.4f}")
