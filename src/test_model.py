import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_model(filename):
    """
    Loads a trained model from a file.
    
    Args:
        filename: The name of the file to load the model from.
        
    Returns:
        The loaded model.
    """
    return joblib.load(filename)

def display_sample_prediction(model, X, y):
    """
    Displays a random sample digit from the dataset and the model's prediction.
    
    Args:
        model: The trained model.
        X: Feature data.
        y: Target data.
    """
    # Select a random sample from the dataset
    sample_index = np.random.randint(0, len(X))
    sample_digit = X[sample_index]
    sample_label = y[sample_index]
    
    # The data is already scaled, so we need to unscale it for visualization
    # This requires the original scaler, which we don't have here.
    # We will reshape and display the scaled data. It won't be a perfect image.
    sample_digit_image = sample_digit.reshape(28, 28)
    print(sample_digit)
    # Get the model's prediction
    prediction = model.predict([sample_digit])
    
    # Display the sample and the prediction
    plt.imshow(sample_digit_image, cmap="binary")
    plt.title(f"Prediction: {prediction[0]}, Actual: {sample_label}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Load the trained model
    model = load_model("mnist_mlp_model.joblib")
    print("Model loaded successfully.")

    # Load the test data for a quick test
    # We need to load the original data to get the unscaled version for visualization
    X_raw, y_raw = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    _, X_test_raw, _, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # We also need the scaled data to make predictions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    _, X_test_scaled, _, _ = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)
    
    # Display a sample prediction
    print("Displaying a random prediction from the test set:")
    display_sample_prediction(model, X_test_scaled, y_test_raw)
