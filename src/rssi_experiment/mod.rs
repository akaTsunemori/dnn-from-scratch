use dnn_from_scratch::loss::Loss;
use dnn_from_scratch::neural_network::NeuralNetwork;
use dnn_from_scratch::report::ReportData;
use dnn_from_scratch::utils::Regression;
use nd::Array2;

mod dataset_setup;
mod plot;

pub trait Training {
    fn train(
        &mut self,
        x_train: Array2<f64>,
        y_train: Array2<f64>,
        x_test: Array2<f64>,
        y_test: Array2<f64>,
        n_epochs: u32,
        initial_learning_rate: f64,
        decay: f64,
    );
}

impl Training for NeuralNetwork {
    fn train(
        &mut self,
        x_train: Array2<f64>,
        y_train: Array2<f64>,
        x_test: Array2<f64>,
        y_test: Array2<f64>,
        n_epochs: u32,
        initial_learning_rate: f64,
        decay: f64,
    ) {
        let mut learning_rate;
        let mut report_data = ReportData::new(n_epochs, "error");
        let loss = Loss::new("mse");
        let error = Loss::new("rmse");
        for epoch in 1..=n_epochs {
            // Training pipeline
            let output = self.forward(&x_train);
            let train_loss = loss.mse(&output, &y_train);
            let train_accuracy = error.rmse(&output, &y_train);
            let scaling_factor = 1. / output.shape()[0] as f64;
            let output_gradient = (&output - &y_train) * scaling_factor;
            learning_rate = initial_learning_rate / (1. + decay * epoch as f64); // Learning rate step decay
            self.backward(&output_gradient, learning_rate, epoch);
            // Testing pipeline
            let output = self.forward(&x_test);
            let test_loss = loss.mse(&output, &y_test);
            let test_accuracy = error.rmse(&output, &y_test);
            // Report
            report_data.add(train_loss, train_accuracy, test_loss, test_accuracy);
            report_data.print_report(epoch);
        }
        // Save training history and plot CDF
        report_data.save_report("rssi_experiment_training_history.txt");
        let predictions = self.forward(&x_test);
        let (sorted_errors, cdf) = Regression::cumulative_distribution(&predictions, &y_test);
        plot::plot_cdf(sorted_errors, cdf, "output/rssi_experiment_cdf.png");
    }
}

pub fn run_rssi_experiment(random_seed: Option<u64>) {
    // RSSI Architecture
    const INPUT_SIZE: usize = 13;
    const OUTPUT_SIZE: usize = 2;
    const HIDDEN_SIZES: [usize; 2] = [256, 256];
    // Load dataset
    let (x_train, y_train, x_test, y_test) =
        dataset_setup::load_rssi_dataset("assets/rssi/rssi-dataset.csv", 0.15);
    println!("RSSI dataset successfully loaded");
    // Neural Network pipeline
    let mut neural_network = NeuralNetwork::new(random_seed);
    neural_network.add_layer(INPUT_SIZE, HIDDEN_SIZES[0], "relu");
    neural_network.add_layer(HIDDEN_SIZES[0], HIDDEN_SIZES[1], "relu");
    neural_network.add_layer(HIDDEN_SIZES[1], OUTPUT_SIZE, "relu");
    neural_network.train(x_train, y_train, x_test, y_test, 2500, 0.0005, 0.00001);
}
