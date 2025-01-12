//! # Report Module
//!
//! This module is responsible for storing and processing training and testing history during machine
//! learning model training. It supports saving training history to a file, printing a formatted report
//! to the console, and tracking error metrics such as accuracy or error. The report includes metrics
//! like training loss, testing loss, training error, and testing error for each epoch.
//!
//! ## Functions
//!
//! - **`add`**: Adds a new set of training and testing metrics (loss and error) for the current epoch.
//! - **`get_errors`**: Returns the vectors of training and testing errors.
//! - **`get_losses`**: Returns the vectors of training and testing losses.
//! - **`is_empty`**: Checks if the report data is empty (i.e., no metrics have been added).
//! - **`print_report`**: Prints a formatted report to the console for the current epoch, showing
//! metrics for training and testing data.
//! - **`save_training_history`**: Saves the training history (loss and error) to a specified output file.
//! - **`save_report`**: Saves the training history to a file in a predefined directory and prints a
//! confirmation message to the console.

use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;

enum ErrorMetric {
    Accuracy,
    Error,
}

/// Struct to store and manage training and testing data for each epoch during model training.
pub struct ReportData {
    n_epochs: u32,          // Number of epochs
    train_losses: Vec<f64>, // Training losses for each epoch
    train_errors: Vec<f64>, // Training errors for each epoch
    test_losses: Vec<f64>,  // Testing losses for each epoch
    test_errors: Vec<f64>,  // Testing errors for each epoch
    metric: ErrorMetric,    // The error metric type (accuracy or error)
}

impl ReportData {
    /// Creates a new `ReportData` object to store training and testing history.
    ///
    /// # Arguments
    /// - `n_epochs`: The total number of epochs for training.
    /// - `error_metric`: The type of error metric to track, either "accuracy" or "error".
    ///
    /// # Returns
    /// Returns a new `ReportData` instance.
    pub fn new(n_epochs: u32, error_metric: &str) -> ReportData {
        let metric = match error_metric {
            "accuracy" => ErrorMetric::Accuracy,
            "error" => ErrorMetric::Error,
            _ => panic!("Unintended error metric passed."),
        };
        ReportData {
            n_epochs,
            train_losses: Vec::new(),
            train_errors: Vec::new(),
            test_losses: Vec::new(),
            test_errors: Vec::new(),
            metric,
        }
    }

    /// Adds a new set of metrics (train loss, train error, test loss, test error) for the current epoch.
    ///
    /// # Arguments
    /// - `train_loss`: The loss value for the training data in the current epoch.
    /// - `train_error`: The error value for the training data in the current epoch.
    /// - `test_loss`: The loss value for the test data in the current epoch.
    /// - `test_error`: The error value for the test data in the current epoch.
    pub fn add(&mut self, train_loss: f64, train_error: f64, test_loss: f64, test_error: f64) {
        self.train_losses.push(train_loss);
        self.train_errors.push(train_error);
        self.test_losses.push(test_loss);
        self.test_errors.push(test_error);
    }

    /// Retrieves the errors (training and testing errors) from the report.
    ///
    /// # Returns
    /// Returns a tuple containing two vectors:
    /// - Training errors as `Vec<f64>`.
    /// - Testing errors as `Vec<f64>`.
    pub fn get_errors(&self) -> (Vec<f64>, Vec<f64>) {
        (self.train_errors.clone(), self.test_errors.clone())
    }

    /// Retrieves the losses (training and testing losses) from the report.
    ///
    /// # Returns
    /// Returns a tuple containing two vectors:
    /// - Training losses as `Vec<f64>`.
    /// - Testing losses as `Vec<f64>`.
    pub fn get_losses(&self) -> (Vec<f64>, Vec<f64>) {
        (self.train_losses.clone(), self.test_losses.clone())
    }

    /// Checks if the `ReportData` is empty (i.e., no data has been added yet).
    ///
    /// # Returns
    /// Returns `true` if the report is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.train_losses.is_empty()
    }

    /// Prints a formatted report for the current epoch, including training and testing loss/error/accuracy.
    ///
    /// The report displays metrics in a table format, where the training and testing losses, as well
    /// as errors/accuracies, are printed for the current epoch.
    ///
    /// # Arguments
    /// - `epoch`: The current epoch number to display.
    pub fn print_report(&self, epoch: u32) {
        assert!(!self.is_empty(), "Error: report_data is empty");
        let n_epochs = self.n_epochs;
        let train_loss = self.train_losses.last().unwrap().to_owned();
        let train_error = self.train_errors.last().unwrap().to_owned();
        let test_loss = self.test_losses.last().unwrap().to_owned();
        let test_error = self.test_errors.last().unwrap().to_owned();
        if epoch > 1 {
            println!("\r\x1b[6A");
        }
        let report_message = match self.metric {
            ErrorMetric::Accuracy => {
                format!(
                    "\
                ┌───────────┬────────────────────────────────┬────────────────────────────────┐  \n\
                │   Epoch   │            Train               │             Test               │  \n\
                ├───────────┼──────────────┬─────────────────┼──────────────┬─────────────────┤  \n\
                │ {:4}/{:<4} │ Loss:{:7.4} │ Accuracy:{:5.1}% │ Loss:{:7.4} │ Accuracy:{:5.1}% │  \n\
                └───────────┴──────────────┴─────────────────┴──────────────┴─────────────────┘  ",
                    epoch,
                    n_epochs,
                    train_loss,
                    train_error * 100.,
                    test_loss,
                    test_error * 100.
                )
            }
            ErrorMetric::Error => {
                format!(
                    "\
                ┌───────────┬────────────────────────────────┬────────────────────────────────┐  \n\
                │   Epoch   │            Train               │             Test               │  \n\
                ├───────────┼──────────────┬─────────────────┼──────────────┬─────────────────┤  \n\
                │ {:4}/{:<4} │ Loss:{:7.2} │ Error:{:9.2} │ Loss:{:7.2} │ Error:{:9.2} │  \n\
                └───────────┴──────────────┴─────────────────┴──────────────┴─────────────────┘  ",
                    epoch, n_epochs, train_loss, train_error, test_loss, test_error
                )
            }
        };
        println!("{}", report_message);
    }

    /// Saves the entire training history (losses and errors) to a specified file.
    ///
    /// The file is created if it doesn't exist, and new entries are appended.
    ///
    /// # Arguments
    /// - `output_path`: The path to the output file where the training history will be saved.
    pub fn save_training_history(&self, output_path: &str) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .expect("Failure when saving training history.");
        for i in 0..self.n_epochs as usize {
            let n_epochs = self.n_epochs;
            let train_loss = self.train_losses[i];
            let train_error = self.train_errors[i];
            let test_loss = self.test_losses[i];
            let test_error = self.test_errors[i];
            let metric = match self.metric {
                ErrorMetric::Accuracy => "Accuracy",
                ErrorMetric::Error => "Error",
            };
            writeln!(
                file,
                "Epoch {}/{} \
                | Train: Loss {:.8}, {} {:.8} \
                | Test: Loss {:.8}, {} {:.8}",
                i + 1,
                n_epochs,
                train_loss,
                metric,
                train_error,
                test_loss,
                metric,
                test_error
            )
            .expect("Failure when saving training history.");
        }
    }

    /// Saves the training history and report to a file within a specific directory (`./output`).
    ///
    /// # Arguments
    /// - `output_file`: The name of the output file where the training report will be saved.
    pub fn save_report(&self, output_file: &str) {
        create_dir_all("./output").expect("Failure saving report.");
        let output_file: &str = &format!("output/{}", output_file);
        self.save_training_history(output_file);
        println!("Training history saved to: {}", output_file);
    }
}
