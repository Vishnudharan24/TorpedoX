# TorpedoX: Cryptographic Algorithm Identification Tool

**TorpedoX** is an AI/ML-based tool designed to identify cryptographic algorithms from a given dataset. It supports two key approaches: 
1. **Class A**: The classification of classical ciphers (specifically ACA ciphers)
2. **Class B**: The analysis of Modern ciphers.

## Overview

TorpedoX provides an efficient solution for automatically identifying cryptographic algorithms by leveraging machine learning techniques. This tool is ideal for cryptanalysis, data security assessments, and cryptographic research.

### 1. Classical Cipher Detection (ACA Ciphers)

For the detection and classification of classical ciphers, **TorpedoX** uses a Feed-Forward Neural Network (FFN) model combined with feature extraction techniques. This approach is specifically tailored to identify ACA (American Cryptogram Association) ciphers, such as:
- Vigenère
- Playfair
- Amsco
- Autokey, and others.

**Feature Extraction:** The model uses a Feed-Forward Neural Network (FFN) to extract distinct features from cipher texts, enabling it to differentiate between various classical cipher types based on their unique recurring characteristics.
### 2. Modern Cipher Analysis

The analysis of modern cryptographic algorithms is handled through three distinct models:

- **Differential Analysis:** This model converts the cipher text into blocks of 16 and performs mathematical and statistical analysis between adjacent blocks. The goal is to detect relationships between blocks, which can reveal underlying cryptographic structures.
  
- **Higher Order Analysis:** This model expands the analysis across multiple adjacent blocks (from 2 to 8 blocks), uncovering more complex patterns in the ciphertext, which might be missed by simpler methods.
  
- **Linear Analysis:** Through XOR operations between adjacent words, this model detects linear patterns that may hint at specific cryptographic algorithms in use.

### Research and Development

We are actively developing a **CNN-FNN hybrid model** that will incorporate a decision fusion approach to combine the results from the three existing models (Differential, Higher Order, and Linear Analysis). This hybrid model aims to improve detection accuracy by 60% leveraging the strengths of each individual analysis method.
