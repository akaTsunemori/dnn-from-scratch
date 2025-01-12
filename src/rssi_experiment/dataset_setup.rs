//! # RSSI Dataset Setup
//!
//! This module provides functionality for loading and preprocessing an RSSI
//! (Received Signal Strength Indicator) dataset.
//!
//! The module contains two main functions:
//! - `min_max_scale`: Scales the values of a numeric matrix to the range [0, 1] using min-max scaling.
//! - `load_rssi_dataset`: Loads an RSSI dataset from a CSV file, splits it into training and testing
//! sets, and applies min-max scaling to the feature data.

use nd::{s, Array2};
use polars::datatypes::Float64Type;
use polars::prelude::{CsvReadOptions, IndexOrder, SerReader};

/// Scales the values of a numeric matrix to the range [0, 1] using min-max normalization.
///
/// # Arguments
/// - `matrix`: A 2-dimensional array (`Array2<f64>`) of numeric values to be scaled.
///
/// # Returns
/// A 2-dimensional array where each value is scaled to the range [0, 1].
fn min_max_scale(matrix: Array2<f64>) -> Array2<f64> {
    let min = matrix.fold(f64::INFINITY, |a, &b| a.min(b));
    let max = matrix.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    matrix.mapv(|x| (x - min) / (max - min))
}

/// Loads an RSSI dataset from a CSV file, splits it into training and testing sets, and applies
/// min-max scaling to the feature data.
///
/// # Arguments
/// - `path_to_csv`: A string slice representing the file path to the CSV file containing the dataset.
/// - `test_proportion`: A `f64` value representing the proportion of the data to use for the test set.
///   Must be in the range `(0, 1)`.
///
/// # Returns
/// A tuple containing four 2-dimensional arrays:
/// - `x_train`: Training set features.
/// - `y_train`: Training set labels.
/// - `x_test`: Testing set features.
/// - `y_test`: Testing set labels.
///
/// # Panics
/// - If `test_proportion` is not in the range `(0, 1)`.
/// - If the CSV file cannot be read or parsed correctly.
///
/// # Notes
/// - The CSV file must have at least 15 columns where:
///   - Columns 1 and 2 are considered as label columns (`y_matrix`).
///   - Columns 3 to 15 are treated as feature columns (`x_matrix`).
pub fn load_rssi_dataset(
    path_to_csv: &str,
    test_proportion: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    assert!(
        0. < test_proportion && test_proportion < 1.,
        "The test proportion should be in the (0, 1) range."
    );
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path_to_csv.into()))
        .unwrap()
        .finish()
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let df_nrows = df.nrows();
    let mut y_matrix = Array2::zeros((df_nrows, 2));
    let mut x_matrix = Array2::zeros((df_nrows, 13));
    for i in 0..df_nrows {
        for j in 1..=2 {
            y_matrix[[i, j - 1]] = df[[i, j]];
        }
        for j in 3..=15 {
            x_matrix[[i, j - 3]] = df[[i, j]];
        }
    }
    let x_matrix = min_max_scale(x_matrix);
    let num_test = (df_nrows as f64 * test_proportion).round() as usize;
    let num_train = df_nrows - num_test;
    let train_slice = s![0..num_train, ..];
    let test_slice = s![num_train..df_nrows, ..];
    let x_train = x_matrix.slice(train_slice).into_owned();
    let y_train = y_matrix.slice(train_slice).into_owned();
    let x_test = x_matrix.slice(test_slice).into_owned();
    let y_test = y_matrix.slice(test_slice).into_owned();
    (x_train, y_train, x_test, y_test)
}
