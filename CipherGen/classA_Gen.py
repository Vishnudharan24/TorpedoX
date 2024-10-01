import string

# Helper functions for each encryption algorithm

# AMSCO Cipher (Basic columnar transposition cipher)
def encrypt_amsco(plaintext):
    # Simple AMSCO-like encryption (Columnar transposition example)
    columns = 5  # You can change this for a different key length
    return ''.join([plaintext[i::columns] for i in range(columns)])

# Autokey Cipher
def encrypt_autokey(plaintext, key):
    alphabet = string.ascii_uppercase
    key = key.upper() + plaintext.upper()
    ciphertext = ''
    for i, letter in enumerate(plaintext.upper()):
        shift = alphabet.index(key[i % len(key)])
        ciphertext += alphabet[(alphabet.index(letter) + shift) % 26]
    return ciphertext

# Baconian Cipher
def encrypt_baconian(plaintext):
    baconian_dict = {
        'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA', 'F': 'AABAB',
        'G': 'AABBA', 'H': 'AABBB', 'I': 'ABAAA', 'J': 'ABAAB', 'K': 'ABABA', 'L': 'ABABB',
        'M': 'ABBAA', 'N': 'ABBAB', 'O': 'ABBBA', 'P': 'ABBBB', 'Q': 'BAAAA', 'R': 'BAAAB',
        'S': 'BAABA', 'T': 'BAABB', 'U': 'BABAA', 'V': 'BABAB', 'W': 'BABBA', 'X': 'BABBB',
        'Y': 'BBAAA', 'Z': 'BBAAB'
    }
    return ''.join([baconian_dict[char.upper()] for char in plaintext if char.upper() in baconian_dict])

# Bazeries Cipher (combined polyalphabetic and transposition cipher)
def encrypt_bazeries(plaintext, key):
    # Simple version of Bazeries cipher using numeric key
    key = [int(d) for d in key]
    sorted_key = sorted(enumerate(key), key=lambda x: x[1])
    cols = len(key)
    ciphertext = [''] * cols
    for i, char in enumerate(plaintext):
        ciphertext[i % cols] += char
    return ''.join([ciphertext[i[0]] for i in sorted_key])

# Beaufort Cipher
def encrypt_beaufort(plaintext, key):
    alphabet = string.ascii_uppercase
    key = key.upper()
    ciphertext = ''
    for i, letter in enumerate(plaintext.upper()):
        shift = alphabet.index(key[i % len(key)])
        ciphertext += alphabet[(shift - alphabet.index(letter)) % 26]
    return ciphertext

# Vigenere Cipher
def encrypt_vigenere(plaintext, key):
    alphabet = string.ascii_uppercase
    key = key.upper()
    ciphertext = ''
    for i, letter in enumerate(plaintext.upper()):
        shift = alphabet.index(key[i % len(key)])
        ciphertext += alphabet[(alphabet.index(letter) + shift) % 26]
    return ciphertext

# Trifid (Trisquare) Cipher
def encrypt_trisquare(plaintext, key):
    # Placeholder example; a real implementation requires a 3x3x3 grid.
    return ''.join(reversed(plaintext))

# Playfair Cipher
def encrypt_playfair(plaintext, key):
    # Placeholder example; real Playfair needs a 5x5 grid and digraphs.
    return ''.join(reversed(plaintext))

# Scytale Cipher
def encrypt_scytale(plaintext, key):
    key = int(key)
    ciphertext = [''] * key
    for i, letter in enumerate(plaintext):
        ciphertext[i % key] += letter
    return ''.join(ciphertext)

# Main function to prompt user and encrypt
def main():
    algorithms = {
        "AMSCO": encrypt_amsco,
        "AUTOKEY": encrypt_autokey,
        "BACONIAN": encrypt_baconian,
        "BAZERIES": encrypt_bazeries,
        "BEAUFORT": encrypt_beaufort,
        "VIGENERE": encrypt_vigenere,
        "TRISQUARE": encrypt_trisquare,
        "PLAYFAIR": encrypt_playfair,
        "SCYTALE": encrypt_scytale
    }

    print("Available algorithms: AMSCO, AUTOKEY, BACONIAN, BAZERIES, BEAUFORT, VIGENERE, TRISQUARE, PLAYFAIR, SCYTALE")
    
    plaintext = input("Enter the plaintext: ").replace(" ", "").upper()
    algorithm = input("Enter the algorithm to use: ").strip().upper()

    if algorithm not in algorithms:
        print(f"Invalid algorithm! Choose from {', '.join(algorithms.keys())}.")
        return
    
    if algorithm in ["AUTOKEY", "BAZERIES", "BEAUFORT", "VIGENERE", "SCYTALE"]:
        key = input("Enter the key: ").strip().upper()
        ciphertext = algorithms[algorithm](plaintext, key)
    else:
        ciphertext = algorithms[algorithm](plaintext)

    print(f"Ciphertext: {ciphertext}")

if __name__ == "__main__":
    main()
