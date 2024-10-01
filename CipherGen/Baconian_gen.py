def baconian_cipher_encode(text):
    baconian_alphabet = {
        'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB',
        'E': 'ABAAA', 'F': 'ABAAB', 'G': 'ABABA', 'H': 'ABABB',
        'I': 'ABBAA', 'J': 'ABBAB', 'K': 'ABBAA', 'L': 'ABBBA',
        'M': 'BAAAA', 'N': 'BAAAB', 'O': 'BAABA', 'P': 'BAABB',
        'Q': 'BABAA', 'R': 'BABAB', 'S': 'BABBA', 'T': 'BABBB',
        'U': 'BBAAA', 'V': 'BBAAB', 'W': 'BBABA', 'X': 'BBABB',
        'Y': 'BBBAA', 'Z': 'BBBAB'
    }
    
    text = text.upper()
    encoded_text = ''.join(baconian_alphabet.get(c, '') for c in text if c in baconian_alphabet)
    
    return encoded_text

def baconian_cipher_decode(encoded_text):
    baconian_alphabet = {
        'AAAAA': 'A', 'AAAAB': 'B', 'AAABA': 'C', 'AAABB': 'D',
        'ABAAA': 'E', 'ABAAB': 'F', 'ABABA': 'G', 'ABABB': 'H',
        'ABBAA': 'I', 'ABBAB': 'J', 'ABBAA': 'K', 'ABBBA': 'L',
        'BAAAA': 'M', 'BAAAB': 'N', 'BAABA': 'O', 'BAABB': 'P',
        'BABAA': 'Q', 'BABAB': 'R', 'BABBA': 'S', 'BABBB': 'T',
        'BBAAA': 'U', 'BBAAB': 'V', 'BBABA': 'W', 'BBABB': 'X',
        'BBBAA': 'Y', 'BBBAB': 'Z'
    }
    
    # Split the encoded text into chunks of 5
    chunks = [encoded_text[i:i+5] for i in range(0, len(encoded_text), 5)]
    decoded_text = ''.join(baconian_alphabet.get(chunk, '') for chunk in chunks)
    
    return decoded_text

# Example usage
if __name__ == "__main__":
    action = input("Do you want to encode or decode? (e/d): ").strip().lower()
    if action == 'e':
        plaintext = input("Enter the plaintext: ")
        #HELLO
        encoded_text = baconian_cipher_encode(plaintext)
        print(f"Encoded text: {encoded_text}")
    elif action == 'd':
        encoded_text = input("Enter the encoded text: ")
        decoded_text = baconian_cipher_decode(encoded_text)
        print(f"Decoded text: {decoded_text}")
    else:
        print("Invalid action. Please choose 'e' to encode or 'd' to decode.")
