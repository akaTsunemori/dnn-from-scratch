//! Deep Neural Network implementation with training functionality for MNIST digit recognition
//!
//! This module provides the implementation for training the neural network, along with a runner for
//! the MNIST experiment.

extern crate dnn_from_scratch;
use dnn_from_scratch::loss::Loss;
use dnn_from_scratch::neural_network::NeuralNetwork;
use dnn_from_scratch::report::ReportData;
use dnn_from_scratch::utils::Classification;
use nd::Array2;

mod dataset_setup;
mod plot;

/// Defines training behavior for the neural network.
pub trait Training {
    /// Trains a neural network using provided training and test data.
    ///
    /// # Arguments
    ///
    /// - `x_train` - Training input data.
    /// - `y_train` - Training labels.
    /// - `x_test` - Test input data.
    /// - `y_test` - Test labels.
    /// - `n_epochs` - Number of training epochs.
    /// - `initial_learning_rate` - Initial learning rate for optimization.
    /// - `decay` - Learning rate decay factor.
    fn train(
        &mut self,
        x_train: Array2<f64>,
        y_train: Array2<f64>,
        x_test: Array2<f64>,
        y_test: Array2<f64>,
        n_epochs: u32,
        initial_learning_rate: f64,
        decay: f64,
    );
}

impl Training for NeuralNetwork {
    /// Implements neural network training using gradient descent with learning rate decay.
    ///
    /// Uses cross-entropy loss and tracks accuracy metrics during training.
    /// Saves training history and generates performance plots.
    ///
    /// # Training Process
    /// 1. Forward pass through network
    /// 2. Loss computation and accuracy measurement
    /// 3. Gradient calculation with batch scaling
    /// 4. Backward pass with learning rate decay
    /// 5. Test set evaluation
    /// 6. Metrics reporting
    fn train(
        &mut self,
        x_train: Array2<f64>,
        y_train: Array2<f64>,
        x_test: Array2<f64>,
        y_test: Array2<f64>,
        n_epochs: u32,
        initial_learning_rate: f64,
        decay: f64,
    ) {
        let mut learning_rate;
        let mut report_data = ReportData::new(n_epochs, "accuracy");
        let loss = Loss::new("cross_entropy");
        for epoch in 1..=n_epochs {
            // Training pipeline
            let output = self.forward(&x_train);
            let train_loss = loss.compute_loss(&output, &y_train);
            let predicted_labels = Classification::argmax(&output, 1);
            let true_labels = Classification::argmax(&y_train, 1);
            let train_accuracy = Classification::compute_accuracy(
                &predicted_labels.into_dyn(),
                &true_labels.into_dyn(),
            );
            let scaling_factor = 6. / output.shape()[0] as f64;
            let output_gradient = (&output - &y_train) * scaling_factor;
            learning_rate = initial_learning_rate / (1. + decay * epoch as f64); // Step decay
            self.backward(&output_gradient, learning_rate, epoch);
            // Testing pipeline
            let output = self.forward(&x_test);
            let test_loss = loss.compute_loss(&output, &y_test);
            let predicted_labels = Classification::argmax(&output, 1);
            let true_labels = Classification::argmax(&y_test, 1);
            let test_accuracy = Classification::compute_accuracy(
                &predicted_labels.into_dyn(),
                &true_labels.into_dyn(),
            );
            // Report
            report_data.add(train_loss, train_accuracy, test_loss, test_accuracy);
            report_data.print_report(epoch);
        }
        // Save training history and plot losses
        report_data.save_report("mnist_experiment_training_history.txt");
        let (train_accuracy, test_accuracy) = report_data.get_errors();
        plot::plot_error(
            n_epochs,
            train_accuracy,
            test_accuracy,
            "output/mnist_experiment_plot.png",
        );
    }
}

/// Runs the MNIST digit recognition experiment with specified architecture.
///
/// # Arguments
/// - `random_seed` - Optional seed for reproducibility
///
/// # Architecture
/// - Input layer: 784 neurons (flattened 28x28 pixels).
/// - Hidden layers: Two layers of 512 neurons with ReLU activation.
/// - Output layer: 10 neurons with softmax activation.
///
/// # Hyperparameters
/// - Epochs: 100
/// - Initial learning rate: 0.001
/// - Learning rate decay: 0.001
pub fn run_mnist_experiment(random_seed: Option<u64>) {
    // MNIST Architecture
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const HIDDEN_SIZES: [usize; 2] = [512, 512];
    // Load dataset
    let (x_train, y_train, x_test, y_test) = dataset_setup::load_mnist_dataset("assets/mnist");
    println!("MNIST dataset successfully loaded");
    // Neural Network pipeline
    let mut neural_network = NeuralNetwork::new(random_seed);
    neural_network.add_layer(INPUT_SIZE, HIDDEN_SIZES[0], "relu");
    neural_network.add_layer(HIDDEN_SIZES[0], HIDDEN_SIZES[1], "relu");
    neural_network.add_layer(HIDDEN_SIZES[1], OUTPUT_SIZE, "softmax");
    neural_network.train(x_train, y_train, x_test, y_test, 100, 0.001, 0.001);
}
