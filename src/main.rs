//! DNN From Scratch Experiments
//!
//! This file serves as the entry point for running two experiments:
//! - **RSSI Experiment**: A regression experiment that maps Received Signal Strength Indicator (RSSI)
//! to geographical coordinates (X, Y).
//! - **MNIST Experiment**: A classification experiment that involves the MNIST dataset, focusing on
//! digit recognition, a well-known classification problem.
//!
//! The experiments are executed with a fixed random seed to ensure reproducibility across different runs.
//! The main function invokes the experiment logic defined in the
//! `rssi_experiment` and `mnist_experiment` modules.

extern crate ndarray as nd;
extern crate ndarray_npy as npy;

mod mnist_experiment;
mod rssi_experiment;

fn main() {
    let random_seed = Some(42); // For reproducibility
    rssi_experiment::run_rssi_experiment(random_seed);
    mnist_experiment::run_mnist_experiment(random_seed);
}
