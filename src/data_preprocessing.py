import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path="data/mnist_784.npz"):
    """
    Loads the MNIST dataset from a local file or downloads it if not present.
    It then scales and splits the data.
    
    Args:
        data_path (str): The path to save/load the dataset.

    Returns:
        A tuple containing the training and testing data splits: 
        (X_train, X_test, y_train, y_test)
    """
    if os.path.exists(data_path):
        # Load data from local file
        with np.load(data_path, allow_pickle=True) as data:
            X, y = data['X'], data['y']
        print("Loaded MNIST data from local file.")
    else:
        # Download data and save to local file
        print("Downloading MNIST data from OpenML...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez_compressed(data_path, X=X, y=y)
        print(f"Saved MNIST data to {data_path}")

    # Convert labels to integers
    y = y.astype(np.uint8)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data loaded and preprocessed successfully.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
