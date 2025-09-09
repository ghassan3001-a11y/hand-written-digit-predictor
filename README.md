# MNIST Digit Recognition Project

This project is an implementation of a neural network to recognize handwritten digits from the MNIST dataset. The model is built using scikit-learn's Multi-layer Perceptron (MLP) classifier.

## Team Members

| AC.NO | Name | Role | Contributions |
|----|------|------|---------------|
| 1 | Ghassan | Lead Developer | Project setup, model development, and training |

## Installation and Setup

To get started with this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd project-name
    ```

2.  **Install uv:**
    Follow the official documentation at https://docs.astral.sh/uv/ to install `uv`.

3.  **Install dependencies:**
    ```bash
    uv sync
    ```

## Usage

This project uses a command-line interface with three main commands: `train`, `test`, and `app`.

### Training the Model

To train the model, use the `train` command. You can also provide custom hyperparameters.

**Default training:**
```bash
uv run main.py train
```

**Custom training:**
```bash
uv run main.py train --hidden_layer_sizes 50 50 --max_iter 20 --learning_rate_init 0.01
```

**Arguments:**

-   `--hidden_layer_sizes`: The number of neurons in the hidden layers (e.g., `50 50` for two layers with 50 neurons each). Default is `100`.
-   `--max_iter`: The maximum number of iterations. Default is `10`.
-   `--alpha`: L2 penalty (regularization term) parameter. Default is `1e-4`.
-   `--learning_rate_init`: The initial learning rate. Default is `0.1`.

### Testing the Model

To test the model with a random sample from the test set, use the `test` command:

```bash
uv run main.py test
```

This will load the saved model and display a plot with the digit and the model's prediction.

### Drawing Application

To run the interactive drawing application, use the `app` command:

```bash
uv run main.py app
```
This will open a window where you can draw a digit with your mouse. When you close the window, the model will predict the digit you drew.

## Project Structure

-   `data/`: Directory for storing dataset files.
-   `docs/`: Directory for project documentation.
-   `notebooks/`: Contains Jupyter notebooks for data exploration and experimentation.
-   `src/`: Contains the main source code for the project.
    -   `data_preprocessing.py`: Handles loading and preprocessing of the MNIST data.
    -   `main.py`: The main script for training the model.
    -   `test_model.py`: A script to test the trained model.
-   `mnist_mlp_model.joblib`: The saved, trained model.
-   `pyproject.toml`: Project configuration and dependencies for `uv`.
-   `README.md`: This file.

## Model

The model is a Multi-layer Perceptron (MLP) classifier from scikit-learn with one hidden layer of 100 neurons. It is trained for 10 epochs and achieves an accuracy of approximately 95.68% on the test set.