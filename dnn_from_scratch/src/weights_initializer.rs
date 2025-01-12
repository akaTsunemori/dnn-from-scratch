//! # Weights Initializer Module
//!
//! This module defines the `WeightsInitializer` structure and a method for initializing weights in
//! a neural network model. Proper weight initialization is essential for faster convergence during
//! training and to avoid issues like vanishing or exploding gradients.
//!
//! ## Supported Initialization Methods
//! - **He Initialization**: A method designed for ReLU and its variants, which uses a scaled normal
//! distribution to initialize weights. The scaling factor is determined by the number of input units
//! to the layer.

use nd::Array2;
use rand::{self, rngs::StdRng, thread_rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};

/// The `WeightsInitializer` structure provides methods for initializing model weights.
pub struct WeightsInitializer {}

impl WeightsInitializer {
    /// He initialization for weights, commonly used with ReLU activation functions.
    ///
    /// This method initializes weights using a normal distribution with a mean of 0 and a standard
    /// deviation determined by the input size. The scaling factor used for the standard deviation is
    /// calculated as `sqrt(2 / input_size)`, which helps avoid vanishing/exploding gradients during
    /// training of neural networks.
    ///
    /// # Arguments
    /// - `input_size`: The number of input units (neurons) in the layer.
    /// - `output_size`: The number of output units (neurons) in the layer.
    /// - `random_seed`: An optional seed for the random number generator. If not provided, a random
    /// seed is generated based on the system's entropy source.
    ///
    /// # Returns
    /// Returns a 2D array of shape `(input_size, output_size)` with weights initialized using
    /// HE-Initialization.
    pub fn he_initialization(
        input_size: usize,
        output_size: usize,
        random_seed: Option<u64>,
    ) -> Array2<f64> {
        // Scaling factor for He initialization: sqrt(2 / input_size)
        let scale = (2. / input_size as f64).sqrt();
        // Normal distribution with mean 0 and standard deviation `scale`
        let normal = Normal::new(0., scale).unwrap();
        // Random number generator, using a seeded RNG if provided
        let mut rng: Box<dyn RngCore> = match random_seed {
            Some(seed_value) => Box::new(StdRng::seed_from_u64(seed_value)),
            None => Box::new(thread_rng()),
        };
        // Generate weights using the normal distribution
        Array2::from_shape_fn((input_size, output_size), |_| normal.sample(&mut rng))
    }
}
