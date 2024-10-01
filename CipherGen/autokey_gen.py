def autokey_cipher(text, key):
    # Ensure key is all uppercase and remove non-alphabetic characters from text
    key = key.upper()
    text = ''.join(filter(str.isalpha, text)).upper()

    # Generate the full key
    full_key = key + text
    full_key = full_key[:len(text)]

    # Perform the encryption
    encrypted_text = []
    for t, k in zip(text, full_key):
        # Calculate the cipher letter
        encrypted_letter = chr(((ord(t) - ord('A') + ord(k) - ord('A')) % 26) + ord('A'))
        encrypted_text.append(encrypted_letter)

    return ''.join(encrypted_text)

# Example usage
if __name__ == "__main__":
    plaintext = input("Enter the plaintext: ")
    #ATTACKATDAWN
    key = input("Enter the key: ")
    encrypted_text = autokey_cipher(plaintext, key)
    print(f"Encrypted text: {encrypted_text}")
