extern crate ndarray as nd;
extern crate ndarray_npy as npy;

mod mnist_experiment;
mod rssi_experiment;

fn main() {
    let random_seed = Some(42); // For reproducibility
    rssi_experiment::run_rssi_experiment(random_seed);
    mnist_experiment::run_mnist_experiment(random_seed);
}
