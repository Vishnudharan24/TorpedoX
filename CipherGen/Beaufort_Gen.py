def beaufort_cipher_encrypt_decrypt(text, key, mode='encrypt'):
    text = text.upper()
    key = key.upper()
    key_repeated = (key * (len(text) // len(key) + 1))[:len(text)]
    
    # Beaufort cipher encryption/decryption process
    def beaufort_operation(t, k):
        return chr((ord(k) - ord(t) + 26) % 26 + ord('A'))
    
    result = []
    for t, k in zip(text, key_repeated):
        if t.isalpha():
            if mode == 'encrypt':
                result.append(beaufort_operation(t, k))
            elif mode == 'decrypt':
                result.append(beaufort_operation(k, t))
        else:
            result.append(t)
    
    return ''.join(result)

# Example usage
if __name__ == "__main__":
    action = input("Do you want to encrypt or decrypt? (e/d): ").strip().lower()
    if action not in ['e', 'd']:
        print("Invalid action. Please choose 'e' to encrypt or 'd' to decrypt.")
        exit()

    plaintext = input("Enter the plaintext: ").upper()
    #HELLO
    key = input("Enter the key: ").upper()
    
    if action == 'e':
        encrypted_text = beaufort_cipher_encrypt_decrypt(plaintext, key, mode='encrypt')
        print(f"Encrypted text: {encrypted_text}")
    elif action == 'd':
        ciphertext = input("Enter the ciphertext: ").upper()
        decrypted_text = beaufort_cipher_encrypt_decrypt(ciphertext, key, mode='decrypt')
        print(f"Decrypted text: {decrypted_text}")
