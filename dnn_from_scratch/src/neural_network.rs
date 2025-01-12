//! # Neural Network Module
//!
//! This module defines the structure and functionality of a simple feedforward neural network. The
//! network supports adding layers, performing forward and backward propagation, and using various
//! activation functions. Each layer of the network is a fully connected layer with an activation
//! function applied to it.
//!
//! ## Main Components:
//! - **`NeuralNetwork` struct**: Manages the layers, random seed, and orchestrates forward and backward
//! passes through the network.
//! - **`add_layer`**: Adds a fully connected layer to the network with a specified input size, output size,
//! and activation function.
//! - **`forward`**: Performs a forward pass through the network, producing the final output.
//! - **`backward`**: Performs backpropagation and adjusts the weights based on the gradients, learning rate,
//! and time step.

use crate::activation::Activation;
use crate::fully_connected::FullyConnected;
use nd::Array2;

/// Represents a feedforward neural network.
pub struct NeuralNetwork {
    layers: Vec<FullyConnected>, // A vector to store all layers of the network.
    random_seed: Option<u64>,    // An optional random seed for weight initialization.
}

impl NeuralNetwork {
    /// Creates a new instance of the `NeuralNetwork`.
    ///
    /// # Arguments
    /// - `random_seed`: An optional seed for random number generation used for weight initialization.
    ///
    /// # Returns
    /// A new `NeuralNetwork` instance with no layers added.
    pub fn new(random_seed: Option<u64>) -> NeuralNetwork {
        NeuralNetwork {
            layers: Vec::new(),
            random_seed,
        }
    }

    /// Adds a new fully connected layer to the network.
    ///
    /// # Arguments
    /// - `input_size`: The number of input nodes (features) for the layer.
    /// - `output_size`: The number of output nodes for the layer.
    /// - `activation`: The activation function to use for the layer (e.g., "relu", "sigmoid").
    pub fn add_layer(&mut self, input_size: usize, output_size: usize, activation: &str) {
        let activation = Activation::new(activation);
        let new_layer = FullyConnected::new(input_size, output_size, activation, self.random_seed);
        self.layers.push(new_layer); // Add the layer to the network
    }

    /// Performs a forward pass through the network, passing the input through each layer.
    ///
    /// # Arguments
    /// - `inputs`: The input data to the network, as a 2D array.
    ///
    /// # Returns
    /// The final output after passing through all layers of the network.
    pub fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        // Perform forward pass starting with the first layer
        let mut layers_output = self.layers[0].forward(inputs.to_owned());
        // Propagate the output through the remaining layers
        for i in 1..self.layers.len() {
            layers_output = self.layers[i].forward(layers_output);
        }
        layers_output // Final output after all layers
    }

    /// Performs backpropagation through the network and adjusts weights using gradients.
    ///
    /// # Arguments
    /// - `gradient`: The gradient of the loss function with respect to the output.
    /// - `learning_rate`: The rate at which the weights are updated.
    /// - `time_step`: The current time step (used for momentum in gradient descent).
    pub fn backward(&mut self, gradient: &Array2<f64>, learning_rate: f64, time_step: u32) {
        // Start backpropagation from the last layer
        let mut layers_gradient =
            self.layers
                .last_mut()
                .unwrap()
                .backward(gradient.to_owned(), learning_rate, time_step);
        // Propagate the gradients backward through each layer
        let range = (0..self.layers.len() - 1).rev(); // Iterate over the layers in reverse
        for i in range {
            layers_gradient = self.layers[i].backward(layers_gradient, learning_rate, time_step);
        }
    }
}
