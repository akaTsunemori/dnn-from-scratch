//! # Optimizer Module
//!
//! This module defines the `Optimizer` structure for managing and applying optimization algorithms
//! in machine learning models. Optimizers adjust the weights and biases of a model during training
//! to minimize the loss function, improving model performance over iterations.
//!
//! ## Supported Optimizers
//! - **Adam**: Adaptive Moment Estimation, a popular optimization algorithm that combines the benefits
//! of Momentum and RMSProp, providing fast convergence and stable updates.
//!
//! ## Features
//! - Initialization of optimizer with configurable parameters such as `beta_1`, `beta_2`, and `epsilon`.
//! - Automatic handling of moment estimates (`m` and `v`) for both weights and biases.
//! - Updating weights and biases based on gradients (`d_weights` and `d_biases`) during backpropagation.

use nd::{Array1, Array2};

/// Enum representing the type of optimizer.
enum OptimizerType {
    /// Adam optimizer.
    Adam,
}

/// The `Optimizer` structure provides tools for updating model parameters during training.
pub struct Optimizer {
    /// The specific optimizer type (e.g., Adam).
    optimizer_type: OptimizerType,
    /// First moment estimate for weights.
    m_weights: Array2<f64>,
    /// Second moment estimate for weights.
    v_weights: Array2<f64>,
    /// First moment estimate for biases.
    m_biases: Array1<f64>,
    /// Second moment estimate for biases.
    v_biases: Array1<f64>,
    /// Exponential decay rate for the first moment estimate.
    beta_1: f64,
    /// Exponential decay rate for the second moment estimate.
    beta_2: f64,
    /// A small constant to prevent division by zero.
    epsilon: f64,
}

impl Optimizer {
    /// Creates a new `Optimizer` instance with the specified configuration.
    ///
    /// # Arguments
    /// - `input_size`: Number of input features (used to initialize weight dimensions).
    /// - `output_size`: Number of output features (used to initialize weight and bias dimensions).
    /// - `optimizer_type`: Type of optimizer to use. Currently supports:
    ///   - `"adam"`
    /// - `beta_1`: Optional, decay rate for the first moment estimate (default: 0.9).
    /// - `beta_2`: Optional, decay rate for the second moment estimate (default: 0.999).
    /// - `epsilon`: Optional, small constant to avoid division by zero (default: 1e-8).
    ///
    /// # Panics
    /// Panics if an unsupported optimizer type is provided.
    pub fn new(
        input_size: usize,
        output_size: usize,
        optimizer_type: &str,
        beta_1: Option<f64>,
        beta_2: Option<f64>,
        epsilon: Option<f64>,
    ) -> Optimizer {
        let optimizer_type = match optimizer_type {
            "adam" => OptimizerType::Adam,
            _ => panic!("Invalid optimizer type passed"),
        };
        let beta_1 = beta_1.unwrap_or(0.9);
        let beta_2 = beta_2.unwrap_or(0.999);
        let epsilon = epsilon.unwrap_or(1e-8);
        Optimizer {
            optimizer_type,
            m_weights: Array2::zeros((input_size, output_size)),
            v_weights: Array2::zeros((input_size, output_size)),
            m_biases: Array1::zeros(output_size),
            v_biases: Array1::zeros(output_size),
            beta_1,
            beta_2,
            epsilon,
        }
    }

    /// Updates the weights and biases using the gradients and the optimizer algorithm.
    ///
    /// # Arguments
    /// - `weights`: Mutable reference to the weight matrix (`Array2<f64>`).
    /// - `biases`: Mutable reference to the bias vector (`Array1<f64>`).
    /// - `d_weights`: Gradient of the loss with respect to weights.
    /// - `d_biases`: Gradient of the loss with respect to biases.
    /// - `learning_rate`: Learning rate for the optimizer.
    /// - `t`: Current time step or iteration (used for bias correction).
    pub fn update(
        &mut self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        d_weights: &Array2<f64>,
        d_biases: &Array1<f64>,
        learning_rate: f64,
        t: u32,
    ) {
        match self.optimizer_type {
            OptimizerType::Adam => {
                self.adam(weights, biases, d_weights, d_biases, learning_rate, t)
            }
        }
    }

    /// Implements the Adam optimization algorithm for updating weights and biases.
    fn adam(
        &mut self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        d_weights: &Array2<f64>,
        d_biases: &Array1<f64>,
        learning_rate: f64,
        t: u32,
    ) {
        let beta_1_factor = 1. - self.beta_1;
        let beta_2_factor = 1. - self.beta_2;
        let beta_1_factor_p = 1. - self.beta_1.powi(t as i32);
        let beta_2_factor_p = 1. - self.beta_2.powi(t as i32);
        // Update weights
        self.m_weights = &(&self.m_weights * self.beta_1) + &(d_weights * beta_1_factor);
        let d_weights_sq = d_weights.mapv(|x| x * x);
        self.v_weights = &(&self.v_weights * self.beta_2) + &d_weights_sq * beta_2_factor;
        let m_hat_weights = &self.m_weights / beta_1_factor_p;
        let v_hat_weights = &self.v_weights / beta_2_factor_p;
        let v_hat_weights_sqrt = v_hat_weights.mapv(|x| x.sqrt() + self.epsilon);
        *weights -= &(learning_rate * &m_hat_weights / &v_hat_weights_sqrt);
        // Update biases
        self.m_biases = &(&self.m_biases * self.beta_1) + &(d_biases * beta_1_factor);
        let d_biases_sq = d_biases.mapv(|x| x * x);
        self.v_biases = &(&self.v_biases * self.beta_2) + &d_biases_sq * beta_2_factor;
        let m_hat_biases = &self.m_biases / beta_1_factor_p;
        let v_hat_biases = &self.v_biases / beta_2_factor_p;
        let v_has_biases_sqrt = v_hat_biases.mapv(|x| x.sqrt() + self.epsilon);
        *biases -= &(learning_rate * &m_hat_biases / &v_has_biases_sqrt);
    }
}
