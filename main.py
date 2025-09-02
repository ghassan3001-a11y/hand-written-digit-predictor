import argparse
import sys
import os
import pygame
import numpy as np
from PIL import Image

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from train_model import train_model, evaluate_model, save_model
from data_preprocessing import load_and_preprocess_data
from test_model import load_model, display_sample_prediction

def draw_digit():
    pygame.init()
    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption("Draw a digit")
    screen.fill((255, 255, 255))
    
    drawing = False
    last_pos = None
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.image.save(screen, "digit.png")
                pygame.quit()
                return
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_pos = None
            
            if event.type == pygame.MOUSEMOTION and drawing:
                if last_pos is not None:
                    pygame.draw.line(screen, (0, 0, 0), last_pos, event.pos, 20)
                last_pos = event.pos
        
        pygame.display.update()

def preprocess_drawn_digit(image_path="digit.png"):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # Invert colors
    img_array = img_array.flatten()
    return img_array.reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description="MNIST Digit Recognition Project")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[100], help='The number of neurons in the hidden layers.')
    train_parser.add_argument('--max_iter', type=int, default=10, help='The maximum number of iterations.')
    train_parser.add_argument('--alpha', type=float, default=1e-4, help='L2 penalty (regularization term) parameter.')
    train_parser.add_argument('--learning_rate_init', type=float, default=.1, help='The initial learning rate.')

    # Testing command
    test_parser = subparsers.add_parser('test', help='Test the model')

    # App command
    app_parser = subparsers.add_parser('app', help='Run the drawing application')

    args = parser.parse_args()

    if args.command == 'train':
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

    elif args.command == 'test':
        # Load the trained model
        model = load_model("mnist_mlp_model.joblib")
        print("Model loaded successfully.")

        # Load the test data for a quick test
        _, X_test, _, y_test = load_and_preprocess_data()
        
        # Display a sample prediction
        print("Displaying a random prediction from the test set:")
        display_sample_prediction(model, X_test, y_test)

    elif args.command == 'app':
        draw_digit()
        model = load_model("mnist_mlp_model.joblib")
        digit = preprocess_drawn_digit()
        prediction = model.predict(digit)
        print(f"The model predicts you drew a: {prediction[0]}")

if __name__ == "__main__":
    main()
