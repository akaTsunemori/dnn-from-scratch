//! This module provides utilities for loading and preprocessing the MNIST dataset stored in `.npy` format.
//!
//! The module includes functions for reading `.npy` files, converting and reshaping arrays, normalizing
//! image pixel values, and converting labels to a one-hot encoded format. It culminates in the
//! `load_mnist_dataset` function, which loads and processes the MNIST dataset.

use nd::{Array2, Array3, ArrayBase, ArrayD, Axis, Dimension, Ix3};
use npy::ReadNpyExt;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Converts an `ndarray` of any dimension (1D, 2D, or 3D) into a 3D array.
///
/// # Arguments
/// - `array`: An `ArrayBase` of any dimension to be converted to 3D.
///
/// # Returns
/// A 3-dimensional array (`ArrayBase<T, Ix3>`).
///
/// # Panics
/// Panics if the input array has a dimensionality other than 1D, 2D, or 3D.
fn to_array3<T, D>(array: ArrayBase<T, D>) -> ArrayBase<T, Ix3>
where
    T: ndarray::Data,
    D: Dimension,
{
    match array.ndim() {
        1 => array
            .insert_axis(Axis(1))
            .insert_axis(Axis(2))
            .into_dimensionality::<Ix3>()
            .unwrap(),
        2 => array
            .insert_axis(Axis(2))
            .into_dimensionality::<Ix3>()
            .unwrap(),
        3 => array.into_dimensionality::<Ix3>().unwrap(),
        _ => panic!("Unsupported dimensionality: {}", array.ndim()),
    }
}

/// Reads an `.npy` file containing the MNIST dataset and converts it to a 3D array of `f64`.
///
/// # Arguments
/// - `path_to_npy`: Path to the `.npy` file.
///
/// # Returns
/// A 3-dimensional array of type `f64`.
///
/// # Panics
/// Panics if the file cannot be read or parsed.
fn read_mnist_npy(path_to_npy: PathBuf) -> Array3<f64> {
    let reader = File::open(path_to_npy).expect("Failure when reading npy file.");
    let array = ArrayD::<u8>::read_npy(reader).expect("Failure when parsing npy file.");
    let array = array.mapv(|v| v as f64);
    let array = to_array3(array);
    array
}

/// Processes a batch of images by flattening each image into a 1D vector.
///
/// # Arguments
/// - `image_batch`: A 3-dimensional array of image data.
///
/// # Returns
/// A 2-dimensional array where each row corresponds to a flattened image.
fn process_images(image_batch: &Array3<f64>) -> Array2<f64> {
    let nrows = image_batch.shape()[0];
    let ncols = image_batch.shape()[1] * image_batch.shape()[2];
    let reshaped_images = image_batch
        .clone()
        .into_shape_clone((nrows, ncols))
        .unwrap();
    let output = reshaped_images.clone();
    output
}

/// Normalizes pixel values in an image dataset to the range [0, 1].
///
/// # Arguments
/// - `processed_images`: A 2-dimensional array of image data.
///
/// # Returns
/// A 2-dimensional array with normalized pixel values.
fn normalize_images(processed_images: &Array2<f64>) -> Array2<f64> {
    let normalized_images = processed_images.mapv(|v| v / 255.);
    normalized_images
}

/// Converts a matrix of labels into a one-hot encoded format.
///
/// # Arguments
/// - `x_train`: A 2-dimensional array of labels (each row contains a single label).
///
/// # Returns
/// A 2-dimensional array where each row is a one-hot encoded vector.
fn to_categorical(x_train: &Array2<f64>) -> Array2<f64> {
    let num_classes = x_train.fold(0.0, |acc: f64, &x| acc.max(x)) as usize + 1;
    let mut result = Array2::zeros((x_train.nrows(), num_classes));
    for (i, row) in x_train.outer_iter().enumerate() {
        if let Some(&label) = row.iter().next() {
            if label >= 0.0 && label < num_classes as f64 {
                result[[i, label as usize]] = 1.0;
            }
        }
    }
    result
}

/// Loads and preprocesses the MNIST dataset from a folder containing `.npy` files.
///
/// The folder should contain the following files:
/// - `x_train.npy`: Training images.
/// - `y_train.npy`: Training labels.
/// - `x_test.npy`: Testing images.
/// - `y_test.npy`: Testing labels.
///
/// # Arguments
/// - `path_to_folder`: The path to the folder containing the `.npy` files.
///
/// # Returns
/// A tuple containing:
/// - `x_train`: Normalized training images (2D array).
/// - `y_train`: One-hot encoded training labels (2D array).
/// - `x_test`: Normalized testing images (2D array).
/// - `y_test`: One-hot encoded testing labels (2D array).
///
/// # Panics
/// Panics if any file cannot be read or processed.
pub fn load_mnist_dataset(
    path_to_folder: &str,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let files = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"];
    let path_to_folder = Path::new(path_to_folder);
    let mut arrays = [None, None, None, None];
    for (i, &file) in files.iter().enumerate() {
        let file = Path::new(file);
        let filepath = path_to_folder.join(file);
        let array = read_mnist_npy(filepath);
        let processed_array = process_images(&array);
        if (i as i32) % 2 == 0 {
            let normalized_array = normalize_images(&processed_array);
            arrays[i] = Some(normalized_array);
        } else {
            let categorized_array = to_categorical(&processed_array);
            arrays[i] = Some(categorized_array);
        }
    }
    let (x_train, y_train, x_test, y_test) = (
        arrays[0].take().unwrap(),
        arrays[1].take().unwrap(),
        arrays[2].take().unwrap(),
        arrays[3].take().unwrap(),
    );
    (x_train, y_train, x_test, y_test)
}
