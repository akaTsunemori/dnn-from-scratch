use nd::Array2;
use rand::{self, rngs::StdRng, thread_rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};

pub struct WeightsInitializer {}

impl WeightsInitializer {
    pub fn he_initialization(
        input_size: usize,
        output_size: usize,
        random_seed: Option<u64>,
    ) -> Array2<f64> {
        let scale = (2. / input_size as f64).sqrt();
        let normal = Normal::new(0., scale).unwrap();
        let mut rng: Box<dyn RngCore> = match random_seed {
            Some(seed_value) => Box::new(StdRng::seed_from_u64(seed_value)),
            None => Box::new(thread_rng()),
        };
        Array2::from_shape_fn((input_size, output_size), |_| normal.sample(&mut rng))
    }
}
