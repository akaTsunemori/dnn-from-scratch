use crate::activation::Activation;
use crate::optimizer::Optimizer;
use crate::weights_initializer::WeightsInitializer;
use ndarray::{Array1, Array2, Axis};

pub struct FullyConnected {
    // Weights and biases
    weights: Array2<f64>,
    biases: Array1<f64>,
    // Activation-related
    activation: Activation,
    // Optimizer-related
    optimizer: Optimizer,
    // Input and Output (for the forward pass)
    input: Array2<f64>,
    output: Array2<f64>,
}

impl FullyConnected {
    /// Initialize a fully-connected layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        random_seed: Option<u64>,
    ) -> FullyConnected {
        let optimizer = Optimizer::new(input_size, output_size, "adam", None, None, None);
        FullyConnected {
            weights: WeightsInitializer::he_initialization(input_size, output_size, random_seed),
            biases: Array1::zeros(output_size),
            activation,
            optimizer,
            input: Array2::zeros((1, 1)),
            output: Array2::zeros((1, 1)),
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        self.input = x;
        let mut z = self.input.dot(&self.weights);
        // Add biases to each row
        z.rows_mut()
            .into_iter()
            .for_each(|mut row| row += &self.biases);
        self.output = self.activation.forward(z);
        self.output.clone()
    }

    /// Backpropagation
    pub fn backward(&mut self, d_values: Array2<f64>, learning_rate: f64, t: u32) -> Array2<f64> {
        // Calculate the derivative of the activation function
        let d_values = self.activation.backward(&d_values, &self.output);
        // Calculate the derivative with respect to weights and biases
        let d_weights = self.input.t().dot(&d_values);
        let d_biases = d_values.sum_axis(Axis(0));
        // Clip derivatives to avoid extreme values
        let d_weights_clipped = d_weights.mapv(|x| x.max(-1.).min(1.));
        let d_biases_clipped = d_biases.mapv(|x| x.max(-1.).min(1.));
        // Calculate gradient with respect to the input
        let d_inputs = d_values.dot(&self.weights.t());
        // Update weights and biases using the optimizer
        self.optimizer.update(
            &mut self.weights,
            &mut self.biases,
            &d_weights_clipped,
            &d_biases_clipped,
            learning_rate,
            t,
        );
        d_inputs
    }
}
