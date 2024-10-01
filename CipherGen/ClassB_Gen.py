from Crypto.Cipher import AES, DES, Blowfish, DES3, ARC4
from Crypto.Cipher import Salsa20
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import scrypt
import base64

# Helper functions for each encryption algorithm

# AES Encryption
def encrypt_aes(plaintext, key):
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

# DES Encryption
def encrypt_des(plaintext, key):
    cipher = DES.new(key, DES.MODE_ECB)
    padded_text = pad(plaintext, DES.block_size)
    ciphertext = cipher.encrypt(padded_text.encode())
    return base64.b64encode(ciphertext).decode()

# Blowfish Encryption
def encrypt_blowfish(plaintext, key):
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    padded_text = pad(plaintext, Blowfish.block_size)
    ciphertext = cipher.encrypt(padded_text.encode())
    return base64.b64encode(ciphertext).decode()

# Twofish (pycryptodome doesn't have direct support; use pycryptodomex)
from Cryptodome.Cipher import Twofish
def encrypt_twofish(plaintext, key):
    cipher = Twofish.new(key, Twofish.MODE_ECB)
    padded_text = pad(plaintext, Twofish.block_size)
    ciphertext = cipher.encrypt(padded_text.encode())
    return base64.b64encode(ciphertext).decode()

# IDEA Encryption
from pycryptodome_idea import IDEA
def encrypt_idea(plaintext, key):
    cipher = IDEA.new(key, IDEA.MODE_ECB)
    padded_text = pad(plaintext, IDEA.block_size)
    ciphertext = cipher.encrypt(padded_text.encode())
    return base64.b64encode(ciphertext).decode()

# RC4 Encryption
def encrypt_rc4(plaintext, key):
    cipher = ARC4.new(key)
    ciphertext = cipher.encrypt(plaintext.encode())
    return base64.b64encode(ciphertext).decode()

# Salsa20 Encryption
def encrypt_salsa20(plaintext, key):
    cipher = Salsa20.new(key=key)
    ciphertext = cipher.encrypt(plaintext.encode())
    return base64.b64encode(cipher.nonce + ciphertext).decode()

# Triple DES Encryption (3DES)
def encrypt_triple_des(plaintext, key):
    cipher = DES3.new(key, DES3.MODE_ECB)
    padded_text = pad(plaintext, DES3.block_size)
    ciphertext = cipher.encrypt(padded_text.encode())
    return base64.b64encode(ciphertext).decode()

# Padding for block ciphers
def pad(text, block_size):
    pad_len = block_size - len(text) % block_size
    return text + chr(pad_len) * pad_len

# Key Derivation for some algorithms
def derive_key(password, key_size):
    salt = get_random_bytes(16)
    key = scrypt(password, salt, key_size, N=2**14, r=8, p=1)
    return key

# Main function to prompt user and encrypt
def main():
    algorithms = {
        "AES": encrypt_aes,
        "DES": encrypt_des,
        "BLOWFISH": encrypt_blowfish,
        "TWOFISH": encrypt_twofish,
        "IDEA": encrypt_idea,
        "RC4": encrypt_rc4,
        "SALSA20": encrypt_salsa20,
        "TRIPLE DES": encrypt_triple_des
    }

    print("Available algorithms: AES, DES, BLOWFISH, TWOFISH, IDEA, RC4, SALSA20, TRIPLE DES")
    
    plaintext = input("Enter the plaintext: ")
    algorithm = input("Enter the algorithm to use: ").strip().upper()

    if algorithm not in algorithms:
        print(f"Invalid algorithm! Choose from {', '.join(algorithms.keys())}.")
        return
    
    key = input("Enter the key (for AES, DES, Blowfish, etc. in hexadecimal or plaintext format): ").strip()
    
    if algorithm == "AES":
        key = derive_key(key.encode(), 16)  # AES-128
    elif algorithm == "DES":
        key = key[:8].encode()  # DES requires 8-byte key
    elif algorithm == "BLOWFISH":
        key = key[:16].encode()  # Blowfish key size can vary
    elif algorithm == "TWOFISH":
        key = key[:16].encode()  # Twofish block size of 16 bytes
    elif algorithm == "IDEA":
        key = key[:16].encode()  # IDEA requires 16-byte key
    elif algorithm == "RC4":
        key = key[:16].encode()  # RC4 key size can vary
    elif algorithm == "SALSA20":
        key = key[:32].encode()  # Salsa20 requires a 32-byte key
    elif algorithm == "TRIPLE DES":
        key = DES3.adjust_key_parity(key[:24].encode())  # Triple DES requires a 24-byte key

    ciphertext = algorithms[algorithm](plaintext, key)

    print(f"Ciphertext: {ciphertext}")

if __name__ == "__main__":
    main()
