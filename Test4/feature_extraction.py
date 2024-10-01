import csv
import numpy as np
import pandas as pd
import string
from collections import Counter
import math

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

def entropy(ciphertext):
    freq = character_frequency(ciphertext)
    N = len(ciphertext)
    return -sum(f/N * math.log2(f/N) for f in freq.values() if f > 0)

def repeated_substrings(ciphertext):
    return [ciphertext[i:i+3] for i in range(len(ciphertext) - 3) if ciphertext.count(ciphertext[i:i+3]) > 1]

def ciphertext_length(ciphertext):
    return len(ciphertext)

def average_distance(ciphertext):
    substrings = repeated_substrings(ciphertext)
    distances = []
    for substring in substrings:
        indices = [i for i, x in enumerate(ciphertext) if x == substring]
        distances.extend([j - i for i in indices for j in indices if i < j])
    return np.mean(distances) if distances else 0

def longest_distance(ciphertext):
    substrings = repeated_substrings(ciphertext)
    max_distance = 0
    for substring in substrings:
        indices = [i for i, x in enumerate(ciphertext) if x == substring]
        if len(indices) > 1:
            max_distance = max(max(j - i for i in indices for j in indices if i < j), max_distance)
    return max_distance

def ioc_26_letters(ciphertext):
    freq = character_frequency(ciphertext)
    N = len(ciphertext)
    return sum(f * (f - 1) for f in freq.values()) / (N * (N - 1))

def calculate_histogram(ciphertext):
    freq = character_frequency(ciphertext)
    return [freq.get(char, 0) for char in string.ascii_uppercase]

def calculate_digrams(ciphertext):
    bigram_counts = bigram_frequency(ciphertext)
    all_bigrams = [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    return [bigram_counts.get(bigram, 0) for bigram in all_bigrams]

def calculate_meta_histogram_features(ciphertext):
    hist = calculate_histogram(ciphertext)
    return [np.mean(hist), np.std(hist)]

def meta_features(ciphertext):
    return [entropy(ciphertext), ioc_26_letters(ciphertext), ciphertext_length(ciphertext)]

def purple_features(ciphertext):
    freq = character_frequency(ciphertext)
    total_letters = sum(freq.values())
    return [freq.get(char, 0) / total_letters for char in string.ascii_uppercase]

def period_ioc_test(ciphertext):
    def period_ioc(ciphertext, period):
        text_len = len(ciphertext)
        blocks = [ciphertext[i:i+period] for i in range(0, text_len, period)]
        block_iocs = [index_of_coincidence(block) for block in blocks if len(block) > 1]
        return np.mean(block_iocs) if block_iocs else 0

    period_iocs = [period_ioc(ciphertext, i) for i in range(1, min(len(ciphertext), 21))]
    return np.max(period_iocs) if period_iocs else 0

def calculate_frequencies(numbers, n):
    freq = Counter(numbers)
    return [freq.get(i, 0) for i in range(n)]

def calculate_chi_square(frequencies):
    expected_freq = sum(frequencies) / len(frequencies)
    return sum(((f - expected_freq) ** 2) / expected_freq for f in frequencies)

def calculate_entropy(numbers):
    freq = Counter(numbers)
    total = len(numbers)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())

def calculate_maximum_index_of_coincidence(numbers):
    freq = Counter(numbers)
    total = len(numbers)
    return sum(count * (count - 1) for count in freq.values()) / (total * (total - 1))

def calculate_max_kappa(numbers):
    freq = Counter(numbers)
    total = len(numbers)
    return sum((count / total) * (1 - count / total) for count in freq.values())

def calculate_log_digraph_score(numbers):
    digrams = [numbers[i:i+2] for i in range(len(numbers) - 1)]
    freq = Counter(tuple(digram) for digram in digrams)
    return np.mean([math.log2(freq.get(digram, 1) + 1) for digram in freq])

def calculate_reverse_log_digraph_score(numbers):
    digrams = [numbers[i:i+2] for i in range(len(numbers) - 1)]
    freq = Counter(tuple(digram) for digram in digrams)
    return np.mean([math.log2(freq.get(digram[::-1], 1) + 1) for digram in freq])

def calculate_digraphic_index_of_coincidence(numbers):
    digrams = [numbers[i:i+2] for i in range(len(numbers) - 1)]
    freq = Counter(tuple(digram) for digram in digrams)
    N = len(digrams)
    return sum(v * (v - 1) for v in freq.values()) / (N * (N - 1))

def has_letter_j(numbers):
    return int(9 in numbers)

def pattern_repetitions(numbers):
    patterns = [tuple(numbers[i:i+3]) for i in range(len(numbers) - 2)]
    freq = Counter(patterns)
    return sum(count - 1 for count in freq.values() if count > 1)

def has_hash(numbers):
    return int(35 in numbers)  # Assuming '#' is represented by 35

def has_space(numbers):
    return int(32 in numbers)  # Assuming ' ' is represented by 32

def has_letter_x(numbers):
    return int(23 in numbers)

def has_digit_0(numbers):
    return int(48 in numbers)  # Assuming '0' is represented by 48

def calculate_normal_order(frequencies):
    return np.mean(frequencies)  # Example calculation

def is_dbl(numbers):
    if len(numbers) % 2 == 0:
        for i in range(0, len(numbers) - 1, 2):
            if numbers[i] == numbers[i + 1]:
                return 1
    return 0

def calculate_nic(numbers):
    """Calculates the maximum Nicodemus IC for periods 3-15."""
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
    """Example calculation."""
    return np.mean(numbers)  # Placeholder calculation

def calculate_phic(numbers):
    """Example calculation."""
    return np.std(numbers)  # Placeholder calculation

def calculate_bdi(numbers):
    """Example calculation."""
    return np.median(numbers)  # Placeholder calculation

def extract_features(text, cipher_type):
    features = {
        'cipher_type': cipher_type,
        'frequency': character_frequency(text),
        'bigram_frequency': bigram_frequency(text),
        'trigram_frequency': trigram_frequency(text),
        'index_of_coincidence': index_of_coincidence(text),
        'entropy': entropy(text),
        'repeated_substrings': repeated_substrings(text),
        'ciphertext_length': ciphertext_length(text),
        'average_distance': average_distance(text),
        'longest_distance': longest_distance(text),
        'ioc_26_letters': ioc_26_letters(text),
        'histogram': calculate_histogram(text),
        'digram': calculate_digrams(text),
        'meta_features': meta_features(text),
        'purple_features': purple_features(text),
        'period_ioc_test': period_ioc_test(text),
        'normal_order': calculate_normal_order(list(map(ord, text))),
        'is_dbl': is_dbl(list(map(ord, text))),
        'nic': calculate_nic(list(map(ord, text))),
        'sdd': calculate_sdd(list(map(ord, text))),
        'ptx': calculate_ptx(list(map(ord, text))),
        'phic': calculate_phic(list(map(ord, text))),
        'bdi': calculate_bdi(list(map(ord, text)))
    }
    return features

# Define the output CSV file
output_csv_file = 'features.csv'

# Load your dataset
input_csv_file = 'cipher_dataset.csv'
df = pd.read_csv(input_csv_file, header=None)

# Write to CSV
with open(output_csv_file, 'w', newline='') as csvfile:
    fieldnames = ['cipher_type', 'frequency', 'bigram_frequency', 'trigram_frequency', 'index_of_coincidence',
                  'entropy', 'repeated_substrings', 'ciphertext_length', 'average_distance', 'longest_distance',
                  'ioc_26_letters', 'histogram', 'digram', 'meta_features', 'purple_features',
                  'period_ioc_test', 'normal_order', 'is_dbl', 'nic', 'sdd', 'ptx', 'phic', 'bdi']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for index, row in df.iterrows():
        ciphertext = row[0]
        cipher_type = 'unknown'  # Replace with actual cipher type if available
        features = extract_features(ciphertext, cipher_type)
        writer.writerow(features)
