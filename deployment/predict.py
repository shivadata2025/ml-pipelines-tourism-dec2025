
"""
Prediction module with multiple fallback options
"""
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Global model variable
model = None

def load_model():
    """Load model with multiple fallback strategies"""
    global model

    if model is not None:
        return model

    print("Loading model...")

    # List of possible model locations
    model_locations = [
        # 1. Local files
        "best_model.pkl",
        "model.pkl",
        "../models/best_model.pkl",
        "mlops-tourism-project/models/best_model.pkl",

        # 2. Try to download from Hugging Face (as fallback)
        None  # Will try Hugging Face if local fails
    ]

    for i, location in enumerate(model_locations):
        if location:  # Try local files first
            try:
                if os.path.exists(location):
                    model = joblib.load(location)
                    print(f"Model loaded from: {location}")
                    return model
            except:
                continue

    # If local files failed, try Hugging Face
    try:
        from huggingface_hub import hf_hub_download
        print("Trying Hugging Face Hub...")
        model_path = hf_hub_download(
            repo_id="krish129/tourism-customer-model",
            filename="best_model.pkl"
        )
        model = joblib.load(model_path)
        print("Model loaded from Hugging Face Hub")
        return model
    except Exception as e:
        print(f"Could not load from Hugging Face: {e}")

    # Last resort: create dummy model
    print("Creating dummy model for demo...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Fit with dummy data
    X_dummy = pd.DataFrame({
        'Age': [25, 35, 45, 55, 65],
        'MonthlyIncome': [20000, 30000, 40000, 50000, 60000]
    })
    y_dummy = [0, 1, 0, 1, 0]
    model.fit(X_dummy, y_dummy)

    print("Dummy model created for demo")
    return model

# Load model when module is imported
model = load_model()

# Define expected columns based on your training
EXPECTED_COLUMNS = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched',
    'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
    'Designation', 'MonthlyIncome'
]

def encode_categorical(df):
    """Encode categorical variables"""
    df_encoded = df.copy()

    # Mapping for categorical variables
    categorical_maps = {
        'TypeofContact': {'Company Invited': 1, 'Self Inquiry': 0},
        'Gender': {'Male': 1, 'Female': 0, 'Fe Male': 0, 'Fe male': 0},
        'Occupation': {'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Free Lancer': 3, 'Business': 1},
        'ProductPitched': {'Basic': 0, 'Deluxe': 1, 'King': 2, 'Standard': 3, 'Super Deluxe': 4},
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2, 'Unmarried': 0},
        'Designation': {'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}
    }

    for col, mapping in categorical_maps.items():
        if col in df_encoded.columns:
            # Convert to string and map
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = df_encoded[col].map(mapping)
            # Fill any NaN with 0
            df_encoded[col] = df_encoded[col].fillna(0).astype(int)

    return df_encoded

def prepare_input(df):
    """Prepare input data for prediction"""
    # Drop unnecessary columns
    cols_to_drop = ['CustomerID', 'ProdTaken']
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Encode categorical variables
    df_encoded = encode_categorical(df_clean)

    # Ensure all expected columns are present
    for col in EXPECTED_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns
    df_encoded = df_encoded[EXPECTED_COLUMNS]

    # Convert all to numeric
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')
    df_encoded = df_encoded.fillna(0)

    return df_encoded

def predict(data_dict: dict):
    """
    Accepts a python dict of input fields and returns model prediction.
    Returns: (prediction, confidence)
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])

        # Prepare input
        df_processed = prepare_input(df)

        # Ensure model is loaded
        if model is None:
            load_model()

        # Make prediction
        prediction = model.predict(df_processed)[0]

        # Try to get probability
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df_processed)[0]
                confidence = proba[1] if prediction == 1 else proba[0]
            else:
                confidence = 0.5
        except:
            confidence = 0.5

        return int(prediction), float(confidence)

    except Exception as e:
        print(f"Prediction error: {e}")

        # Fallback: simple rule-based prediction
        age = data_dict.get('Age', 35)
        income = data_dict.get('MonthlyIncome', 20000)
        passport = data_dict.get('Passport', 0)

        # Simple rules
        score = 0
        if age < 40: score += 1
        if income > 25000: score += 1
        if passport == 1: score += 1

        prediction = 1 if score >= 2 else 0
        confidence = 0.7 if prediction == 1 else 0.3

        return prediction, confidence

# For testing
if __name__ == "__main__":
    # Test data
    test_data = {
        "CustomerID": 1001,
        "ProdTaken": 0,
        "Age": 35.0,
        "TypeofContact": "Company Invited",
        "CityTier": 2,
        "DurationOfPitch": 15.0,
        "Occupation": "Salaried",
        "Gender": "Male",
        "NumberOfPersonVisiting": 2,
        "NumberOfFollowups": 3.0,
        "ProductPitched": "Deluxe",
        "PreferredPropertyStar": 4.0,
        "MaritalStatus": "Married",
        "NumberOfTrips": 2.0,
        "Passport": 1,
        "PitchSatisfactionScore": 4,
        "OwnCar": 1,
        "NumberOfChildrenVisiting": 0.0,
        "Designation": "Manager",
        "MonthlyIncome": 25000.0
    }

    print("Testing predict function...")
    pred, conf = predict(test_data)
    print(f"Prediction: {pred} (1=Buy, 0=Not Buy)")
    print(f"Confidence: {conf:.1%}")
