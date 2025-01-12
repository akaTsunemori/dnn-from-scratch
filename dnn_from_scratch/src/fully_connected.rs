//! # Fully-Connected Module
//!
//! This module defines a fully-connected (dense) layer for neural networks, providing
//! functionality for forward propagation, backpropagation, and parameter updates. It integrates
//! with activation functions, optimizers, and weight initializers to form a complete trainable layer.
//!
//! ## Features
//! - **Weights and Biases Initialization**: Supports He initialization for weights and zeros for biases.
//! - **Activation Functions**: Configurable activation functions via the `Activation` module.
//! - **Optimizers**: Leverages the `Optimizer` module for efficient parameter updates.
//! - **Forward and Backward Pass**: Implements feedforward and backpropagation mechanisms for training.

use crate::activation::Activation;
use crate::optimizer::Optimizer;
use crate::weights_initializer::WeightsInitializer;
use ndarray::{Array1, Array2, Axis};

/// Represents a fully-connected (dense) layer in a neural network.
pub struct FullyConnected {
    /// Weight matrix connecting input and output neurons.
    weights: Array2<f64>,
    /// Bias vector for the output neurons.
    biases: Array1<f64>,
    /// Activation function applied to the layer's output.
    activation: Activation,
    /// Optimizer used to update weights and biases.
    optimizer: Optimizer,
    /// Input data from the forward pass.
    input: Array2<f64>,
    /// Output data from the forward pass.
    output: Array2<f64>,
}

impl FullyConnected {
    /// Creates a new fully-connected layer with specified parameters.
    ///
    /// # Arguments
    /// - `input_size`: Number of input features (neurons).
    /// - `output_size`: Number of output features (neurons).
    /// - `activation`: The activation function to use for this layer.
    /// - `random_seed`: Optional seed for random weight initialization.
    ///
    /// # Returns
    /// A new `FullyConnected` instance with initialized weights, biases, and optimizer.
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        random_seed: Option<u64>,
    ) -> FullyConnected {
        let optimizer = Optimizer::new(input_size, output_size, "adam", None, None, None);
        FullyConnected {
            weights: WeightsInitializer::he_initialization(input_size, output_size, random_seed),
            biases: Array1::zeros(output_size),
            activation,
            optimizer,
            input: Array2::zeros((1, 1)),
            output: Array2::zeros((1, 1)),
        }
    }

    /// Performs the forward pass through the layer.
    ///
    /// # Arguments
    /// - `x`: A 2D array (`Array2<f64>`) containing the input data. Each row corresponds to a sample.
    ///
    /// # Returns
    /// A 2D array representing the layer's output after applying weights, biases, and activation.
    pub fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        self.input = x;
        let mut z = self.input.dot(&self.weights);
        // Add biases to each row
        z.rows_mut()
            .into_iter()
            .for_each(|mut row| row += &self.biases);
        self.output = self.activation.forward(z);
        self.output.clone()
    }

    /// Performs backpropagation for this layer.
    ///
    /// # Arguments
    /// - `d_values`: Gradients of the loss with respect to the layer's output.
    /// - `learning_rate`: Learning rate for the optimizer.
    /// - `t`: The current training step or iteration (used by some optimizers).
    ///
    /// # Returns
    /// Gradients of the loss with respect to the layer's input.
    ///
    /// # Details
    /// - Computes gradients with respect to weights, biases, and inputs.
    /// - Updates weights and biases using the optimizer.
    /// - Clips gradients to avoid extreme values.
    pub fn backward(&mut self, d_values: Array2<f64>, learning_rate: f64, t: u32) -> Array2<f64> {
        // Calculate the derivative of the activation function
        let d_values = self.activation.backward(&d_values, &self.output);
        // Calculate the derivative with respect to weights and biases
        let d_weights = self.input.t().dot(&d_values);
        let d_biases = d_values.sum_axis(Axis(0));
        // Clip derivatives to avoid extreme values
        let d_weights_clipped = d_weights.mapv(|x| x.max(-1.).min(1.));
        let d_biases_clipped = d_biases.mapv(|x| x.max(-1.).min(1.));
        // Calculate gradient with respect to the input
        let d_inputs = d_values.dot(&self.weights.t());
        // Update weights and biases using the optimizer
        self.optimizer.update(
            &mut self.weights,
            &mut self.biases,
            &d_weights_clipped,
            &d_biases_clipped,
            learning_rate,
            t,
        );
        d_inputs
    }
}
