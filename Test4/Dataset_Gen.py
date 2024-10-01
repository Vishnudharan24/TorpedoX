import random
import string
import csv

# Cipher generation functions...

# AMSCO Cipher
def generate_amsco_cipher(plaintext, period=4):
    rows = [''] * period
    i = 0
    for index, char in enumerate(plaintext):
        rows[i] += char
        if index % 2 == 0:
            i = (i + 1) % period
        else:
            i = (i + 2) % period
    return ''.join(rows)

# Autokey Cipher
def generate_autokey_cipher(plaintext, key):
    key = key.upper()
    ciphertext = []
    key_index = 0
    for char in plaintext:
        if char.isalpha():
            shift = ord(key[key_index]) - ord('A')
            cipher_char = chr(((ord(char) - ord('A') + shift) % 26) + ord('A'))
            ciphertext.append(cipher_char)
            key_index = (key_index + 1) % len(key)
        else:
            ciphertext.append(char)
    return ''.join(ciphertext)

# Baconian Cipher
def generate_bazeries_cipher(plaintext):
    substitution = ''.join(random.sample(string.ascii_uppercase, 26))
    trans_table = str.maketrans(string.ascii_uppercase, substitution)
    return plaintext.upper().translate(trans_table)

# Beaufort Cipher
def generate_beaufort_cipher(plaintext, period=10):
    def beaufort_encrypt_char(c, key_char):
        return chr(((ord(key_char) - ord(c)) % 26) + ord('A'))
    
    key = 'KEY'  # Simple key for demonstration
    ciphertext = []
    key_length = len(key)
    
    for i, char in enumerate(plaintext.upper()):
        if char.isalpha():
            encrypted_char = beaufort_encrypt_char(char, key[i % key_length])
            ciphertext.append(encrypted_char)
        else:
            ciphertext.append(char)
    
    return ''.join(ciphertext)

# Cipher Sample Generation
def generate_cipher_sample(algorithm, length=100):
    plaintext = ''.join(random.choices(string.ascii_uppercase, k=length))
    
    if algorithm == "AMSCO":
        return generate_amsco_cipher(plaintext)
    elif algorithm == "AUTOKEY":
        key = ''.join(random.choices(string.ascii_uppercase, k=10))  # Random key
        return generate_autokey_cipher(plaintext, key)
    elif algorithm == "BAZERIES":
        return generate_bazeries_cipher(plaintext)
    elif algorithm == "BEAUFORT":
        return generate_beaufort_cipher(plaintext)
    else:
        return "UNKNOWN_CIPHER"

# Dataset Generation
def generate_dataset(filename, algorithms, num_samples_per_algo=40000, length=100):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ciphertext', 'algorithm'])  # Header
        for algorithm in algorithms:
            for _ in range(num_samples_per_algo):
                ciphertext = generate_cipher_sample(algorithm, length)
                writer.writerow([ciphertext, algorithm])
    print(f"Dataset saved to {filename}")

# Main Execution
if __name__ == "__main__":
    algorithms = ["AMSCO", "AUTOKEY", "BAZERIES", "BEAUFORT"]
    generate_dataset('cipher_dataset.csv', algorithms, num_samples_per_algo=2500, length=100)
