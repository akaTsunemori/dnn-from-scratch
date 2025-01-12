//! # Loss Module
//!
//! This module defines a `Loss` structure for calculating various loss functions commonly used in machine
//! learning. Loss functions measure the difference between predicted outputs and target values, guiding
//! model optimization.
//!
//! ## Supported Loss Functions
//! - **Cross-Entropy Loss**: Used for classification tasks, measures the divergence between predicted
//! probabilities and actual class labels.
//! - **Mean Squared Error (MSE)**: Commonly used for regression tasks, measures the average squared
//! difference between predicted and actual values.
//! - **Root Mean Squared Error (RMSE)**: Similar to MSE but provides results on the same scale as the
//! data by taking the square root of the MSE.

use nd::Array2;

/// Enum representing the types of loss functions available.
enum LossType {
    /// Cross-Entropy Loss.
    CrossEntropy,
    /// Mean Squared Error (MSE) Loss.
    MSE,
    /// Root Mean Squared Error (RMSE) Loss.
    RMSE,
}

/// The `Loss` structure encapsulates the selected loss function and provides methods to compute the
/// loss between predicted and target values.
pub struct Loss {
    /// The specific loss function used (e.g., CrossEntropy, MSE, RMSE).
    loss: LossType,
}

impl Loss {
    /// Creates a new `Loss` instance with the specified loss type.
    ///
    /// # Arguments
    /// - `loss_type`: A string specifying the type of loss. Valid options are:
    ///   - `"cross_entropy"`
    ///   - `"mse"`
    ///   - `"rmse"`
    ///
    /// # Panics
    /// This function panics if an invalid loss type is provided.
    pub fn new(loss_type: &str) -> Loss {
        match loss_type {
            "cross_entropy" => Loss {
                loss: LossType::CrossEntropy,
            },
            "mse" => Loss {
                loss: LossType::MSE,
            },
            "rmse" => Loss {
                loss: LossType::RMSE,
            },
            _ => panic!("Invalid loss passed."),
        }
    }

    /// Computes the loss value for the given output and target arrays using the selected loss function.
    ///
    /// # Arguments
    /// - `output`: A 2D array (`Array2<f64>`) containing the predicted values.
    /// - `target`: A 2D array (`Array2<f64>`) containing the target values.
    ///
    /// # Returns
    /// A `f64` value representing the computed loss.
    pub fn compute_loss(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        match self.loss {
            LossType::CrossEntropy => self.categorical_cross_entropy(output, target),
            LossType::MSE => self.mse(output, target),
            LossType::RMSE => self.rmse(output, target),
        }
    }

    /// Computes the categorical cross-entropy loss.
    ///
    /// # Arguments
    /// - `output`: A 2D array of predicted probabilities.
    /// - `target`: A 2D array of target probabilities or one-hot encoded class labels.
    ///
    /// # Returns
    /// A `f64` value representing the categorical cross-entropy loss.
    ///
    /// # Details
    /// Adds a small constant (epsilon) to prevent numerical instability when taking logarithms.
    pub fn categorical_cross_entropy(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let mut total_sum = 0.;
        let mut total_elements = 0.;
        const EPSILON: f64 = 1e-10;
        output
            .iter()
            .zip(target.iter())
            .for_each(|(predicted, expected)| {
                total_sum += *predicted * f64::ln(*expected + EPSILON);
                total_elements += 1.;
            });
        -1. * (total_sum / total_elements)
    }

    /// Computes the mean squared error (MSE) loss.
    ///
    /// # Arguments
    /// - `output`: A 2D array of predicted values.
    /// - `target`: A 2D array of target values.
    ///
    /// # Returns
    /// A `f64` value representing the mean squared error.
    pub fn mse(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let diff = output - target;
        let squared_diff = diff.mapv(|x| x.powi(2));
        let mse = squared_diff.mean().unwrap_or(0.0);
        mse
    }

    /// Computes the root mean squared error (RMSE) loss.
    ///
    /// # Arguments
    /// - `output`: A 2D array of predicted values.
    /// - `target`: A 2D array of target values.
    ///
    /// # Returns
    /// A `f64` value representing the root mean squared error.
    pub fn rmse(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let diff = output - target;
        let squared_diff = diff.mapv(|x| x.powi(2));
        let mse = squared_diff.mean().unwrap_or(0.0);
        mse.sqrt()
    }
}
