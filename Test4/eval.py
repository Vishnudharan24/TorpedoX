import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from train import extract_features  # Import your feature extraction function

def load_resources():
    """Load the scaler, label encoder, and model."""
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    model = load_model('cipher_model.h5')
    return scaler, label_encoder, model

def preprocess_input(ciphertext, scaler):
    """Extract features from ciphertext, pad if necessary, and scale them."""
    features = extract_features(ciphertext)
    
    # Print feature size for debugging
    print(f"Extracted features size: {len(features)}")
    
    feature_df = pd.DataFrame([features])
    
    # Get the expected number of features from the scaler
    expected_features = scaler.n_features_in_
    
    # Check if the number of features is less than expected
    if len(features) < expected_features:
        # Pad with zeros if there are fewer features
        features = np.pad(features, (0, expected_features - len(features)), 'constant')
    elif len(features) > expected_features:
        # Truncate if there are more features (this is just a safety check)
        features = features[:expected_features]
    
    # Convert features to DataFrame and scale
    feature_df = pd.DataFrame([features])
    X_scaled = scaler.transform(feature_df)
    return X_scaled

def predict_algorithm(ciphertext):
    """Predict the algorithm of the given ciphertext."""
    scaler, label_encoder, model = load_resources()
    X_scaled = preprocess_input(ciphertext, scaler)
    
    # Predict the class probabilities
    y_pred_prob = model.predict(X_scaled)
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    
    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(y_pred_class)
    
    return predicted_label[0]

if __name__ == "__main__":
    # Get user input
    ciphertext = input("Enter the ciphertext: ")
    
    # Predict the algorithm
    try:
        predicted_algorithm = predict_algorithm(ciphertext)
        print(f"The predicted cryptographic algorithm is: {predicted_algorithm}")
    except ValueError as e:
        print(f"Error: {e}")
