import random
import string

def generate_bazeries_cipher(plaintext, key):
    # Create a grid for the Bazeries cipher
    grid_size = len(key)
    grid = [''] * grid_size

    # Fill the grid with characters from the plaintext
    for i, char in enumerate(plaintext):
        grid[i % grid_size] += char

    # Order columns by the key
    sorted_key_indices = sorted(range(len(key)), key=lambda k: key[k])
    
    # Generate the ciphertext by reading columns in the order defined by the key
    ciphertext = ''.join(grid[i] for i in sorted_key_indices)
    
    return ciphertext

def generate_bazeries_cipher_key():
    # Create a key for the Bazeries cipher (random permutation of letters)
    return ''.join(random.sample(string.ascii_uppercase, 26))

# Example usage
plaintext = "Rajeshwari"
key = generate_bazeries_cipher_key()  # Generate a random key
ciphertext = generate_bazeries_cipher(plaintext, key)

print(f"Plaintext: {plaintext}")
print(f"Key: {key}")
print(f"Ciphertext: {ciphertext}")
