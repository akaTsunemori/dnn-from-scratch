extern crate ndarray as nd;
extern crate ndarray_npy as npy;

mod mnist_experiment;
mod rssi_experiment;

fn main() {
    mnist_experiment::run_mnist_experiment();
    rssi_experiment::run_rssi_experiment();
}
