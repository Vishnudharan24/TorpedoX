import math

def amsco_cipher_encrypt(plaintext, key):
    # Prepare plaintext by removing spaces and converting to uppercase
    plaintext = ''.join(plaintext.split()).upper()
    n = len(key)
    
    # Calculate the number of rows needed
    num_rows = math.ceil(len(plaintext) / n)
    
    # Pad plaintext to fit the matrix if needed
    plaintext = plaintext.ljust(num_rows * n, 'X')
    
    # Create the matrix
    matrix = [plaintext[i * n:(i + 1) * n] for i in range(num_rows)]
    
    # Create the key-index mapping
    sorted_key_indices = sorted(range(len(key)), key=lambda x: key[x])
    
    # Reorder columns based on the sorted key
    encrypted_matrix = []
    for row in matrix:
        encrypted_matrix.append(''.join(row[idx] for idx in sorted_key_indices))
    
    return ''.join(encrypted_matrix)

def amsco_cipher_decrypt(ciphertext, key):
    n = len(key)
    
    # Calculate the number of rows
    num_rows = len(ciphertext) // n
    
    # Create the matrix
    matrix = [''] * num_rows
    for i in range(num_rows):
        matrix[i] = ciphertext[i * n:(i + 1) * n]
    
    # Create the key-index mapping
    sorted_key_indices = sorted(range(len(key)), key=lambda x: key[x])
    sorted_key_indices = [sorted_key_indices.index(i) for i in range(len(key))]
    
    # Reorder rows based on the sorted key
    reordered_matrix = [''] * num_rows
    for i in range(num_rows):
        for idx, orig_idx in enumerate(sorted_key_indices):
            reordered_matrix[i] += matrix[i][orig_idx]
    
    # Remove padding and reconstruct plaintext
    plaintext = ''.join(reordered_matrix).rstrip('X')
    
    return plaintext

# Example usage
if __name__ == "__main__":
    action = input("Do you want to encrypt or decrypt? (e/d): ").strip().lower()
    key = input("Enter the key: ").strip().upper()
    
    if action == 'e':
        plaintext = input("Enter the plaintext: ")
        #MEET ME AT THE PARK
        encrypted_text = amsco_cipher_encrypt(plaintext, key)
        print(f"Encrypted text: {encrypted_text}")
    elif action == 'd':
        ciphertext = input("Enter the ciphertext: ")
        decrypted_text = amsco_cipher_decrypt(ciphertext, key)
        print(f"Decrypted text: {decrypted_text}")
    else:
        print("Invalid action. Please choose 'e' to encrypt or 'd' to decrypt.")
