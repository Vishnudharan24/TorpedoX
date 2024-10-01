import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Function to load the trained model and label encoders
def load_model_and_encoders(model_filename, encoder_filename):
    # Load the trained model
    model = load_model(model_filename)
    print(f"Model loaded from {model_filename}")

    # Load the label encoder
    with open(encoder_filename, 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    print(f"Label encoder loaded from {encoder_filename}")
    
    return model, label_encoder

# Function to preprocess the input ciphertext
def preprocess_cipher(ciphertext):
    # Convert characters to ASCII codes
    cipher_data = [ord(c) for c in ciphertext]
    
    # Pad the sequence to match the input shape used during training
    padded_data = pad_sequences([cipher_data], padding='post')  # Ensure correct shape
    
    # Reshape to match the CNN input shape (samples, timesteps, 1)
    padded_data = padded_data.reshape((padded_data.shape[0], padded_data.shape[1], 1))
    
    return padded_data

# Function to predict the algorithm based on the ciphertext
def predict_algorithm(ciphertext, model, label_encoder):
    # Preprocess the ciphertext
    processed_ciphertext = preprocess_cipher(ciphertext)
    
    # Predict the algorithm (returns the predicted label index)
    prediction = model.predict(processed_ciphertext)
    
    # Get the label with the highest probability
    predicted_label = np.argmax(prediction, axis=1)
    
    # Map the numeric label to the algorithm name using the label encoder
    algorithm_name = label_encoder[predicted_label[0]]
    
    return algorithm_name

# Main evaluation function
if __name__ == "__main__":
    # Load the trained model and label encoder
    model, label_encoder = load_model_and_encoders("cipher_model.h5", "label_encoder.pkl")
    
    # Input ciphertext to be evaluated
    input_ciphertext = ""  # Example input ciphertext (replace this with actual ciphertext)
    
    # Predict the algorithm
    predicted_algorithm = predict_algorithm(input_ciphertext, model, label_encoder)
    
    # Display the result
    print(f"The predicted algorithm for the given ciphertext is: {predicted_algorithm}")
