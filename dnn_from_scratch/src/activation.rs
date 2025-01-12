//! # Activation Module
//!
//! This module provides a structure and functionality to handle activation functions commonly used
//! in neural networks. It supports the following activation types:
//!
//! - **ReLU (Rectified Linear Unit)**: Used to introduce non-linearity by outputting the input
//! directly if it is positive; otherwise, it outputs zero.
//! - **Softmax**: Converts a vector of real numbers into a probability distribution, typically
//! used in the output layer for multi-class classification problems.
//! - **None**: A pass-through activation that does not modify the input or output.

use nd::{Array2, Axis};

/// Enum representing supported activation types.
enum ActivationType {
    /// Softmax activation function.
    Softmax,
    /// ReLU activation function.
    ReLU,
    /// No activation function (pass-through).
    None,
}

/// The `Activation` structure manages activation functions and their forward
/// and backward passes.
pub struct Activation {
    activation_type: ActivationType,
}

impl Activation {
    /// Creates a new `Activation` instance based on the specified type.
    ///
    /// # Arguments
    /// - `activation_type`: A string specifying the type of activation.
    ///   Supported values are `"relu"`, `"softmax"`, and `"none"`.
    ///
    /// # Panics
    /// This function will panic if an invalid activation type is provided.
    pub fn new(activation_type: &str) -> Activation {
        let activation_type = match activation_type {
            "relu" => ActivationType::ReLU,
            "softmax" => ActivationType::Softmax,
            "none" => ActivationType::None,
            _ => panic!("Invalid activation type."),
        };
        Activation { activation_type }
    }

    /// Performs the forward pass of the activation function on the input array.
    ///
    /// # Arguments
    /// - `z`: A 2D array (`Array2<f64>`) containing the input data.
    ///
    /// # Returns
    /// A 2D array with the activation function applied element-wise (ReLU) or row-wise (Softmax).
    pub fn forward(&self, mut z: Array2<f64>) -> Array2<f64> {
        match self.activation_type {
            ActivationType::ReLU => z.mapv(|v| v.max(0.)),
            ActivationType::Softmax => {
                for mut row in z.rows_mut() {
                    let z_max = row.iter().fold(f64::NEG_INFINITY, |max, &v| max.max(v));
                    row.iter_mut().for_each(|v| *v -= z_max);
                }
                let mut exp_values = z.mapv(|v| v.exp());
                for mut row in exp_values.rows_mut() {
                    let sum: f64 = row.iter().sum();
                    row.iter_mut().for_each(|v| *v /= sum);
                }
                exp_values
            }
            ActivationType::None => z.to_owned(),
        }
    }

    /// Performs the backward pass (gradient calculation) of the activation function.
    ///
    /// # Arguments
    /// - `d_values`: A 2D array containing gradients propagated from the subsequent layer.
    /// - `z`: A 2D array containing the output of the forward pass (or the input to the activation
    /// function).
    ///
    /// # Returns
    /// A 2D array with gradients adjusted based on the activation function.
    pub fn backward(&self, d_values: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        let mut d_values = d_values.clone();
        match self.activation_type {
            ActivationType::ReLU => d_values *= &z.mapv(|x| if x > 0. { 1. } else { 0. }),
            ActivationType::Softmax => {
                for i in 0..d_values.nrows() {
                    let gradient = d_values.row(i).to_owned();
                    let diagonal = Array2::from_diag(&gradient);
                    let outer_product = gradient
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&gradient.clone().insert_axis(Axis(0)));
                    let jacobian_matrix = diagonal - outer_product;
                    let transformed_gradient =
                        jacobian_matrix.dot(&z.row(i).to_owned().insert_axis(Axis(1)));
                    let result = transformed_gradient.index_axis(Axis(1), 0);
                    d_values.row_mut(i).assign(&result);
                }
            }
            ActivationType::None => {}
        }
        d_values
    }
}
