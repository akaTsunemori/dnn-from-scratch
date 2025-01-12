use nd::{Array1, Array2};

enum OptimizerType {
    Adam,
}

pub struct Optimizer {
    optimizer_type: OptimizerType,
    m_weights: Array2<f64>,
    v_weights: Array2<f64>,
    m_biases: Array1<f64>,
    v_biases: Array1<f64>,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
}

impl Optimizer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        optimizer_type: &str,
        beta_1: Option<f64>,
        beta_2: Option<f64>,
        epsilon: Option<f64>,
    ) -> Optimizer {
        let optimizer_type = match optimizer_type {
            "adam" => OptimizerType::Adam,
            _ => panic!("Invalid optimizer type passed"),
        };
        let beta_1 = beta_1.unwrap_or(0.9);
        let beta_2 = beta_2.unwrap_or(0.999);
        let epsilon = epsilon.unwrap_or(1e-8);
        Optimizer {
            optimizer_type,
            m_weights: Array2::zeros((input_size, output_size)),
            v_weights: Array2::zeros((input_size, output_size)),
            m_biases: Array1::zeros(output_size),
            v_biases: Array1::zeros(output_size),
            beta_1,
            beta_2,
            epsilon,
        }
    }

    pub fn update(
        &mut self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        d_weights: &Array2<f64>,
        d_biases: &Array1<f64>,
        learning_rate: f64,
        t: u32,
    ) {
        match self.optimizer_type {
            OptimizerType::Adam => {
                self.adam(weights, biases, d_weights, d_biases, learning_rate, t)
            }
        }
    }

    fn adam(
        &mut self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        d_weights: &Array2<f64>,
        d_biases: &Array1<f64>,
        learning_rate: f64,
        t: u32,
    ) {
        let beta_1_factor = 1. - self.beta_1;
        let beta_2_factor = 1. - self.beta_2;
        let beta_1_factor_p = 1. - self.beta_1.powi(t as i32);
        let beta_2_factor_p = 1. - self.beta_2.powi(t as i32);
        // Update weights
        self.m_weights = &(&self.m_weights * self.beta_1) + &(d_weights * beta_1_factor);
        let d_weights_sq = d_weights.mapv(|x| x * x);
        self.v_weights = &(&self.v_weights * self.beta_2) + &d_weights_sq * beta_2_factor;
        let m_hat_weights = &self.m_weights / beta_1_factor_p;
        let v_hat_weights = &self.v_weights / beta_2_factor_p;
        let v_hat_weights_sqrt = v_hat_weights.mapv(|x| x.sqrt() + self.epsilon);
        *weights -= &(learning_rate * &m_hat_weights / &v_hat_weights_sqrt);
        // Update biases
        self.m_biases = &(&self.m_biases * self.beta_1) + &(d_biases * beta_1_factor);
        let d_biases_sq = d_biases.mapv(|x| x * x);
        self.v_biases = &(&self.v_biases * self.beta_2) + &d_biases_sq * beta_2_factor;
        let m_hat_biases = &self.m_biases / beta_1_factor_p;
        let v_hat_biases = &self.v_biases / beta_2_factor_p;
        let v_has_biases_sqrt = v_hat_biases.mapv(|x| x.sqrt() + self.epsilon);
        *biases -= &(learning_rate * &m_hat_biases / &v_has_biases_sqrt);
    }
}
