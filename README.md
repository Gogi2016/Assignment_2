# Linear Regression Model for Predicting Y from X

## Introduction

This project involves the development of a **Linear Regression** model that predicts a dependent variable (Y) based on an independent variable (X). The goal is to build a simple regression model using **Rust** and evaluate its performance through training on synthetic data, adjusting parameters over several epochs, and testing the model's predictions.

The model will predict the value of Y for a given value of X based on the learned weight and bias from training data. This project is designed to demonstrate the basic principles of linear regression, how to implement it from scratch, and how to evaluate its performance.

## Problem Statement

The primary challenge is to train a model that can learn the relationship between input data (X) and the output data (Y). Linear regression models are widely used in machine learning for predicting continuous values. In this case, we want to predict a value of Y based on a known relationship to X. Our model will utilize random data with some added noise to simulate a real-world scenario.

## Setup and Requirements

To run this project, ensure that you have the following installed:

- **Rust**: Install the Rust from https://www.rust-lang.org/tools/install abd verify the installation by running rustc –version in your terminal
- **Install Rust Rover IDE**: Download Rust Rover from https://www.jetbrains.com/rust/
- **Cargo**: The Rust package manager is included when you install Rust.

### Steps to Set Up the Project

1. **Clone the repository**:
   
    git clone <>
    cd <repository_directory>

2. **Install dependencies**:
    The project relies on the `rand` crate for generating random numbers. Ensure you have the necessary dependencies by adding the following to the `Cargo.toml` file:
    ```toml
    [dependencies]
    rand = "0.8"
    ```

3. **Run the project**:
    To train the model and see the results, use the following command:
    ```bash
    cargo run
    ```

### Expected Output

The program will output the following:

- Loss values at each epoch during training.
- The trained model’s weight and bias.
- Predictions made by the model for a test value (x = 5).
- A comparison between the actual and predicted values for further evaluation.

## Approach

The linear regression problem was approached as follows:

1. **Data Generation**: 
    - We generate synthetic data with a known linear relationship (Y = aX + b) and introduce random noise to simulate real-world data variability.
    - The independent variable X is randomly sampled within a range of -10.0 to 10.0, and the dependent variable Y is calculated as a linear function of X with added noise.

2. **Model Definition**:
    - The model was defined as a simple linear regression function: `Y = weight * X + bias`, where `weight` and `bias` are parameters to be learned.
    - The loss function used is **Mean Squared Error (MSE)**, which measures the difference between predicted and actual values:
      \[
      MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
      \]
    - The model is trained over multiple epochs by using **gradient descent** to minimize the loss.

3. **Training Process**:
    - The model was trained for 1000 epochs.
    - In each epoch, the weights were adjusted to minimize the loss function, with the learning rate being a hyperparameter influencing the speed of convergence.

4. **Evaluation**:
    - After training, the model’s accuracy was tested using the predicted value of `y` for `x = 5`.
    - The predicted result was compared against the actual data to assess the quality of the model.

## Results and Evaluation

The model was successfully trained over 1000 epochs, and the following results were obtained:

- **Trained Model Parameters**:
  - **Weight**: 1.9774
  - **Bias**: 1.0096
- **Predicted value for x = 5**: `y ≈ 10.89`
- **Actual value for y**: `11.0` (Note: Slight difference due to random noise)

The model was able to approximate the actual data with reasonable accuracy. Over training, the loss decreased significantly, suggesting that the model is learning and adjusting its parameters effectively. Despite some random noise in the dataset, the model captured the underlying linear trend well.

## Challenges

- **Handling Deprecated Functions**: While working on this project, I encountered several warnings related to deprecated functions in the `rand` crate, particularly with `rand::thread_rng` and `rand::Rng::gen_range`. I had to update the code to use the new API to avoid potential issues in future versions of the library.
- **Noise in Data**: The synthetic data includes noise, which sometimes makes the learning process less smooth. The model is limited to simple linear regression and may not handle complex patterns well.
  
## Reflection on the Learning Process

This project provided a hands-on opportunity to implement a fundamental machine learning algorithm—linear regression—in a low-level programming language like Rust. The process allowed me to:

- Deepen my understanding of linear regression and its core components (e.g., loss function, gradient descent, model training).
- Encounter challenges with handling random noise in data and applying standard machine learning techniques.
- Gain exposure to Rust’s data manipulation and handling libraries, specifically the `rand` crate, which I used for data generation.

Additionally, working with the Rust programming language sharpened my ability to implement algorithms in a systems-level language, which is known for its memory safety and performance.

## Resources

- [Rust Documentation](https://doc.rust-lang.org/book/)
- [rand crate documentation](https://docs.rs/rand/latest/rand/)
- [Machine Learning Crash Course - Google](https://developers.google.com/machine-learning/crash-course)
- [Linear Regression - Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
