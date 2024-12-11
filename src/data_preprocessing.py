import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load dataset from the specified file path.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess data (e.g., normalization, encoding categorical variables).
    """
    # Example: Normalize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop('class', axis=1))
    return scaled_data, data['class']