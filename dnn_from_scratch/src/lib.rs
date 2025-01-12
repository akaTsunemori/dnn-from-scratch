//! # Neural Network Library
//!
//! This library provides a simple framework for creating, training, and evaluating neural networks.
//! It includes various modules to handle neural network layers, activation functions, weight initialization,
//! optimizers, loss functions, and training reports.
//!
//! ## Modules
//!
//! - **`activation`**: Contains the definitions and implementations for activation functions used in
//! the neural network layers.
//! - **`fully_connected`**: Defines the fully connected layer (dense layer) of the neural network,
//! implementing both the forward and backward passes.
//! - **`loss`**: Contains various loss functions for training the neural network, such as mean squared
//! error and cross-entropy.
//! - **`neural_network`**: The core module defining the neural network structure, allowing for the
//! addition of layers and the execution of forward and backward propagation.
//! - **`optimizer`**: Contains optimizers for adjusting the network weights during training, such as
//! Adam, to minimize the loss function.
//! - **`report`**: Handles reporting and saving training progress, including accuracy, error, and
//! loss over epochs.
//! - **`utils`**: Utility functions for general tasks like random number generation, logging, etc.
//! - **`weights_initializer`**: Provides weight initialization strategies for the neural network layers,
//! such as HE-Initialization.

extern crate ndarray as nd;
extern crate rand;
extern crate rand_distr;

mod activation; // Module defining activation functions (e.g., ReLU)
mod fully_connected; // Module for the fully connected (dense) layer
pub mod loss; // Module containing loss functions (e.g., MSE, CrossEntropy)
pub mod neural_network; // Core module for building and training the neural network
mod optimizer; // Optimizers (e.g., Adam) for weight updates
pub mod report; // Module for generating training reports and saving history
pub mod utils; // Utility functions for general tasks
mod weights_initializer; // Module for weight initialization strategies like He initialization
