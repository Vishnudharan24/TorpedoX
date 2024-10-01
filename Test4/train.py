import csv
import numpy as np
import pandas as pd
import string
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Feature extraction functions
def character_frequency(ciphertext):
    return Counter(ciphertext)

def bigram_frequency(ciphertext):
    return Counter([ciphertext[i:i+2] for i in range(len(ciphertext) - 1)])

def trigram_frequency(ciphertext):
    return Counter([ciphertext[i:i+3] for i in range(len(ciphertext) - 2)])

def index_of_coincidence(ciphertext):
    freq = character_frequency(ciphertext)
    N = len(ciphertext)
    return sum(f * (f - 1) for f in freq.values()) / (N * (N - 1))

def ioc_26_letters(ciphertext):
    ciphertext = ''.join([c for c in ciphertext.upper() if c in string.ascii_uppercase])
    if len(ciphertext) < 2:
        return 0
    return index_of_coincidence(ciphertext)

def entropy(ciphertext):
    freq = character_frequency(ciphertext)
    N = len(ciphertext)
    return -sum(f/N * math.log2(f/N) for f in freq.values() if f > 0)

def calculate_histogram(ciphertext):
    freq = character_frequency(ciphertext)
    histogram = [freq.get(chr(i), 0) for i in range(256)]  # Histogram for all ASCII characters
    return histogram

def calculate_digrams(ciphertext):
    digram_freq = bigram_frequency(ciphertext)
    digrams = [digram_freq.get(ciphertext[i:i+2], 0) for i in range(len(ciphertext) - 1)]
    
    # Pad or trim to a fixed length (e.g., 256 for consistency)
    fixed_length = 256
    if len(digrams) < fixed_length:
        digrams.extend([0] * (fixed_length - len(digrams)))  # Pad with zeros
    else:
        digrams = digrams[:fixed_length]  # Trim if longer than fixed length
    
    return digrams

def average_distance(ciphertext):
    positions = {char: [] for char in set(ciphertext)}
    for i, char in enumerate(ciphertext):
        positions[char].append(i)
    distances = [positions[char][i+1] - positions[char][i] for char in positions for i in range(len(positions[char])-1)]
    return np.mean(distances) if distances else 0

def longest_distance(ciphertext):
    positions = {char: [] for char in set(ciphertext)}
    for i, char in enumerate(ciphertext):
        positions[char].append(i)
    distances = [positions[char][i+1] - positions[char][i] for char in positions for i in range(len(positions[char])-1)]
    return max(distances) if distances else 0

def calculate_meta_histogram_features(ciphertext):
    return [sum(1 for char in ciphertext if char == chr(i)) for i in range(256)]  # Simplified

def meta_features(ciphertext):
    return [len(ciphertext), len(set(ciphertext))]  # Placeholder for more detailed features

def purple_features(ciphertext):
    return [1 if 'X' in ciphertext else 0]  # Simplified placeholder

def period_ioc_test(ciphertext):
    return index_of_coincidence(ciphertext)  # Placeholder for periodical testing

def calculate_frequencies(numbers, base):
    freq = Counter(numbers)
    return [freq.get(i, 0) for i in range(base)]

def calculate_chi_square(frequencies):
    expected = sum(frequencies) / len(frequencies)
    return sum((f - expected) ** 2 / expected for f in frequencies if expected > 0)

def calculate_entropy(numbers):
    freq = Counter(numbers)
    total = len(numbers)
    return -sum(f / total * math.log2(f / total) for f in freq.values() if f > 0)

def calculate_maximum_index_of_coincidence(numbers):
    return max(numbers) / len(numbers) if numbers else 0

def calculate_max_kappa(numbers):
    return sum(numbers) / len(numbers) if numbers else 0

def calculate_log_digraph_score(numbers):
    return sum(math.log2(numbers[i] + 1) for i in range(len(numbers) - 1))

def calculate_reverse_log_digraph_score(numbers):
    return sum(math.log2(numbers[i] + 1) for i in range(len(numbers) - 1, 0, -1))

def calculate_digraphic_index_of_coincidence(numbers):
    return index_of_coincidence(numbers)

def has_letter_j(numbers):
    return 1 if 74 in numbers else 0  # ASCII for 'J'

def pattern_repetitions(numbers):
    return sum(1 for i in range(len(numbers) - 1) if numbers[i] == numbers[i + 1])

def has_hash(numbers):
    return 1 if 35 in numbers else 0  # ASCII for '#'

def has_space(numbers):
    return 1 if 32 in numbers else 0  # ASCII for space

def has_letter_x(numbers):
    return 1 if 88 in numbers else 0  # ASCII for 'X'

def has_digit_0(numbers):
    return 1 if 48 in numbers else 0  # ASCII for '0'

def calculate_normal_order(frequencies):
    return sorted(frequencies, reverse=True)

def is_dbl(numbers):
    if len(numbers) % 2 == 0:
        for i in range(0, len(numbers) - 1, 2):
            if numbers[i] == numbers[i + 1]:
                return 1
    return 0

def calculate_nic(numbers):
    OUTPUT_ALPHABET = list(range(256))  # Assuming you are working with ASCII values 0-255
    nics = []
    col_len = 5
    for i in range(1, 16):
        ct = [[0] * len(OUTPUT_ALPHABET) for _ in range(16)]
        block_len = len(numbers) // (col_len * i)
        limit = block_len * col_len * i
        index = 0
        for j in range(limit):
            ct[index][numbers[j]] += 1
            if (j + 1) % col_len == 0:
                index = (index + 1) % i
        z = 0
        for j in range(i):
            x = 0
            y = 0
            for k in range(len(OUTPUT_ALPHABET)):
                x += ct[j][k] * (ct[j][k] - 1)
                y += ct[j][k]
            if y > 1:
                z += x / (y * (y - 1))
        z = z / i
        nics.append(z)
    return max(nics[2:])

def calculate_sdd(numbers):
    """Calculate SDD for numbers."""
    sdd_matrix = np.random.rand(26, 26)  # Example: Replace with actual SDD data
    score = 0
    for i in range(len(numbers) - 1):
        if numbers[i] < 26 and numbers[i + 1] < 26:
            score += sdd_matrix[numbers[i]][numbers[i + 1]]
    return score / (len(numbers) - 1) / 10

def calculate_ptx(numbers):
    return np.mean(numbers)  # Placeholder calculation

def calculate_phic(numbers):
    return np.std(numbers)  # Placeholder calculation

def calculate_bdi(numbers):
    return np.median(numbers)  # Placeholder calculation

def repeated_substrings(ciphertext, min_length=3):
    """
    Find repeated substrings of a minimum length.
    """
    substrings = {}
    length = len(ciphertext)
    for l in range(min_length, length // 2 + 1):
        for i in range(length - l + 1):
            substring = ciphertext[i:i + l]
            if substring in substrings:
                substrings[substring] += 1
            else:
                substrings[substring] = 1
    return substrings

def extract_features(text):
    features = {
        'frequency': list(character_frequency(text).values()),
        'bigram_frequency': list(bigram_frequency(text).values()),
        'trigram_frequency': list(trigram_frequency(text).values()),
        'index_of_coincidence': index_of_coincidence(text),
        'entropy': entropy(text),
        'repeated_substrings': list(repeated_substrings(text).values()),
        'ciphertext_length': len(text),
        'average_distance': average_distance(text),
        'longest_distance': longest_distance(text),
        'ioc_26_letters': ioc_26_letters(text),
        'histogram': calculate_histogram(text),
        'digram': calculate_digrams(text),
        'meta_features': meta_features(text),
        'purple_features': purple_features(text),
        'period_ioc_test': period_ioc_test(text)
    }

    # Convert features into a flat list
    feature_vector = []
    for key in sorted(features.keys()):
        value = features[key]
        if isinstance(value, list):
            feature_vector.extend(value)
        else:
            feature_vector.append(value)
    
    return feature_vector

# Load dataset and preprocess
def process_dataset(file_path):
    df = pd.read_csv(file_path)

    # Print columns to debug
    print("Columns in the dataset:", df.columns)

    # Check if 'cipher_type' column exists
    if 'cipher_type' not in df.columns:
        raise ValueError("The 'cipher_type' column is missing from the dataset.")
    
    # Extract features and labels
    feature_list = [extract_features(row['ciphertext']) for index, row in df.iterrows()]
    feature_df = pd.DataFrame(feature_list).fillna(0)  # Handle missing values
    
    X = feature_df
    y = df['cipher_type'].astype('category').cat.codes  # Convert categorical labels to numeric codes

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')  # Save scaler for later use

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, 'label_encoder.pkl')  # Save label encoder for later use

    return X_scaled, y_encoded

# Build and train model
def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset_csv = 'cipher_dataset.csv'  # Replace with your dataset path
    X, y = process_dataset(dataset_csv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1], len(np.unique(y)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred_classes))
    print("Classification Report:\n", classification_report(y_test, y_pred_classes))

    model.save('cipher_model.h5')  # Save the trained model
