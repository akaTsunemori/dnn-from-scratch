//! # Classification and Regression Utility Module
//!
//! This module provides utility functions for classification and regression tasks, commonly found in
//! machine learning models. It includes methods for calculating classification accuracy, performing
//! the argmax operation for classification tasks, and evaluating errors in regression tasks, specifically
//! calculating the cumulative distribution function (CDF) for RMSE (Root Mean Square Error).
//!
//! ## Functions
//!
//! - **`Classification::argmax`**: Returns the indices of the maximum values along a specified axis
//! in a 2D array.
//! - **`Classification::compute_accuracy`**: Calculates the accuracy of the model’s predictions for
//! classification tasks.
//! - **`Regression::cumulative_distribution`**: Computes the cumulative distribution for the errors
//! between predicted and expected values in regression tasks.

use nd::{Array1, Array2, ArrayD};

/// Represents the `Classification` struct, which provides methods related to classification tasks.
pub struct Classification {}

/// Represents the `Regression` struct, which provides methods related to regression tasks.
pub struct Regression {}

impl Classification {
    /// Perform argmax on a 2D array along a given axis (0 for rows, 1 for columns).
    ///
    /// This method returns the indices of the maximum values along the specified axis.
    /// It is commonly used in classification problems to determine the predicted class.
    ///
    /// # Arguments
    /// - `arr`: A 2D array where each row or column represents a set of values.
    /// - `axis`: The axis along which the operation is performed:
    ///   - `0`: Perform argmax along rows.
    ///   - `1`: Perform argmax along columns.
    ///
    /// # Returns
    /// Returns a 1D array containing the indices of the maximum values for each row or column,
    /// depending on the axis.
    pub fn argmax(arr: &Array2<f64>, axis: usize) -> Array1<usize> {
        let shape = arr.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        let (upper_bound_i, upper_bound_j) = match axis {
            0 => (ncols, nrows), // Operate along rows (0)
            1 => (nrows, ncols), // Operate along columns (1)
            _ => panic!("Axis must be 0 or 1"),
        };
        let mut result = Vec::with_capacity(upper_bound_i);
        for i in 0..upper_bound_i {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx = 0;
            for j in 0..upper_bound_j {
                let val = arr[[i, j]];
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            result.push(max_idx);
        }
        Array1::from_vec(result)
    }

    /// Compute the accuracy of a classification model.
    ///
    /// This method compares the output predictions with the target labels and computes the proportion
    /// of matching elements.
    ///
    /// # Arguments
    /// - `output`: A multi-dimensional array representing the predicted values.
    /// - `target`: A multi-dimensional array representing the true target values.
    ///
    /// # Returns
    /// Returns a `f64` value representing the accuracy of the predictions (range 0.0 to 1.0).
    pub fn compute_accuracy<T>(output: &ArrayD<T>, target: &ArrayD<T>) -> f64
    where
        T: PartialEq,
    {
        let mut total_elements = 0;
        let mut equal_elements = 0;
        output
            .iter()
            .zip(target.iter())
            .for_each(|(predicted, expected)| {
                total_elements += 1;
                if *predicted == *expected {
                    equal_elements += 1;
                }
            });
        (equal_elements as f64) / (total_elements as f64)
    }
}

impl Regression {
    /// Calculate the cumulative distribution of errors (RMSE) in a regression task.
    ///
    /// This method computes the RMSE for each data point, sorts the errors, and calculates the cumulative
    /// distribution function (CDF) of the errors.
    ///
    /// # Arguments
    /// - `predictions`: A 2D array representing the predicted values.
    /// - `expected`: A 2D array representing the expected values.
    ///
    /// # Returns
    /// Returns a tuple of two vectors:
    /// - A vector of sorted errors (RMSE).
    /// - A vector of cumulative distribution function (CDF) values for the sorted errors.
    pub fn cumulative_distribution(
        predictions: &Array2<f64>,
        expected: &Array2<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut errors = Vec::new();
        let (mut error_x, mut error_y, mut rmse);
        let top = predictions.shape()[0];
        for i in 0..top {
            error_x = predictions[[i, 0]] - expected[[i, 0]];
            error_y = predictions[[i, 1]] - expected[[i, 1]];
            rmse = (error_x.powi(2) + error_y.powi(2)).sqrt();
            errors.push(rmse);
        }
        // Sort the errors
        let mut sorted_errors = errors.clone();
        sorted_errors.sort_by(|&a, b| a.partial_cmp(b).unwrap());
        // Calculate the CDF
        let len = sorted_errors.len() as f64;
        let cdf = Vec::from_iter((1..=(sorted_errors.len() + 1)).map(|i| i as f64 / len));
        (sorted_errors, cdf)
    }
}
