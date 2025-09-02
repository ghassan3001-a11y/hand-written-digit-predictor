import argparse
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sys
import os
import joblib

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import load_and_preprocess_data

def train_model(X_train, y_train, hidden_layer_sizes, max_iter, alpha, learning_rate_init):
    """
    Trains a Multi-layer Perceptron (MLP) classifier on the training data.
    
    Args:
        X_train: Training feature data.
        y_train: Training target data.
        hidden_layer_sizes (tuple): The number of neurons in the hidden layers.
        max_iter (int): The maximum number of iterations.
        alpha (float): L2 penalty (regularization term) parameter.
        learning_rate_init (float): The initial learning rate.
        
    Returns:
        The trained MLP classifier.
    """
    # Create and train the model
    mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha,
                        solver='adam', verbose=10, random_state=1,
                        learning_rate_init=learning_rate_init, early_stopping=True, n_iter_no_change=5)
    mlp_clf.fit(X_train, y_train)
    return mlp_clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data.
    
    Args:
        model: The trained model.
        X_test: Test feature data.
        y_test: Test target data.
        
    Returns:
        The accuracy of the model on the test data.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_model(model, filename):
    """
    Saves the trained model to a file.
    
    Args:
        model: The trained model.
        filename: The name of the file to save the model to.
    """
    joblib.dump(model, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP classifier on the MNIST dataset.")
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[256, 128], help='The number of neurons in the hidden layers.')
    parser.add_argument('--max_iter', type=int, default=50, help='The maximum number of iterations.')
    parser.add_argument('--alpha', type=float, default=1e-4, help='L2 penalty (regularization term) parameter.')
    parser.add_argument('--learning_rate_init', type=float, default=0.001, help='The initial learning rate.')
    args = parser.parse_args()

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train the model
    model = train_model(X_train, y_train, tuple(args.hidden_layer_sizes), args.max_iter, args.alpha, args.learning_rate_init)
    print("Model trained successfully.")
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model
    save_model(model, "mnist_mlp_model.joblib")
    print("Model saved to mnist_mlp_model.joblib")
