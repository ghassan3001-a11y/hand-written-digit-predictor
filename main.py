import argparse
import sys
import os
import pygame
import numpy as np
from PIL import Image, ImageOps
import joblib
import matplotlib.pyplot as plt
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from train_model import train_model, evaluate_model, save_model
from data_preprocessing import load_and_preprocess_data
from test_model import load_model, display_sample_prediction

def draw_digit():
    PIXEL_SIZE = 5   # Size of each displayed pixel
    GRID_SIZE = 64
    BRUSH_SIZE = 2    # Thickness of the brush in grid cells
    WINDOW_SIZE = PIXEL_SIZE * GRID_SIZE

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Draw a digit (MNIST style)")
    canvas = np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8)

    running = True
    drawing = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                x, y = event.pos
                grid_x = x // PIXEL_SIZE
                grid_y = y // PIXEL_SIZE
                for dx in range(-BRUSH_SIZE//2, BRUSH_SIZE//2 + 1):
                    for dy in range(-BRUSH_SIZE//2, BRUSH_SIZE//2 + 1):
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            canvas[ny, nx] = 0

        # Draw the canvas on the window
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color = (canvas[i, j],) * 3  # White for 0, black for 255
                pygame.draw.rect(screen, color,
                                 (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
        pygame.display.update()

    # Save the digit as an image
    img = Image.fromarray(canvas)
    img.save("digit.png")
    pygame.quit()

def preprocess_drawn_digit(image_path="digit.png"):
    img = Image.open(image_path).convert('L')
    # Invert and fit to 28x28 MNIST size
    img = ImageOps.invert(img)
    img = ImageOps.fit(img, (28, 28), centering=(0.5, 0.5))
    # Convert to numpy array, flatten, normalize
    img_array = np.array(img).astype(np.float32)
    img_array = img_array.flatten().reshape(1, -1)
    
    # Apply scaler if provided
    '''if scaler is not None:
        img_array = scaler.transform(img_array)'''
    return img_array

def main():
    parser = argparse.ArgumentParser(description="MNIST Digit Recognition Project")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[100])
    train_parser.add_argument('--max_iter', type=int, default=10)
    train_parser.add_argument('--alpha', type=float, default=1.0)
    train_parser.add_argument('--learning_rate_init', type=float, default=0.1)

    # Testing command
    subparsers.add_parser('test', help='Test the model')

    # App command
    subparsers.add_parser('app', help='Run the drawing application')

    args = parser.parse_args()

    if args.command == 'train':
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model = train_model(X_train, y_train, tuple(args.hidden_layer_sizes),
                            args.max_iter, args.alpha, args.learning_rate_init)
        print("Model trained successfully.")
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Model accuracy: {accuracy:.4f}")
        save_model(model, "mnist_mlp_model.joblib")
        print("Model saved to mnist_mlp_model.joblib")

    elif args.command == 'test':
        model = load_model("mnist_mlp_model.joblib")
        print("Model loaded successfully.")
        _, X_test, _, y_test = load_and_preprocess_data()
        print("Displaying a random prediction from the test set:")
        display_sample_prediction(model, X_test, y_test)

    elif args.command == 'app':
        draw_digit()
        model = load_model("mnist_mlp_model.joblib")

        digit = preprocess_drawn_digit()
        print(digit)
        prediction = model.predict(digit)
        print(prediction)
        print(f"The model predicts you drew a: {prediction[0]}")

if __name__ == "__main__":
    main()
