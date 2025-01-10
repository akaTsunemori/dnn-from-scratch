# DNN From Scratch

This repository implements a deep neural network (DNN) from scratch in Rust, focusing on two key experiments: image classification with the MNIST dataset and signal strength-based predictions using an RSSI dataset. The project demonstrates building and training a neural network without relying on external machine learning libraries.

---

## 🚀 Features

- **Custom Neural Network Implementation:** Build and train DNNs using only Rust libraries and custom modules.
- **Examples for Two Experiments:**
  - **MNIST Dataset:** Handwritten digit classification.
  - **RSSI Dataset:** Analysis and predictions based on signal strength data.
- **Modular Codebase:** Cleanly separated concerns such as activation functions, loss computation, and neural network architecture.
- **Visualization:** Generate reports and plots to visualize experiment results.

---

## 📂 Directory Structure

```
dnn-from-scratch/
├── Cargo.toml               # Project dependencies and configuration
├── LICENSE                  # License information
├── assets/                  # Datasets and auxiliary data
│   ├── mnist/
│   │   ├── x_test.npy       # MNIST test images
│   │   ├── x_train.npy      # MNIST training images
│   │   ├── y_test.npy       # MNIST test labels
│   │   └── y_train.npy      # MNIST training labels
│   └── rssi/
│       └── rssi-dataset.csv # RSSI dataset
├── dnn_from_scratch/        # Core library for the neural network
│   ├── Cargo.toml           # Library-specific dependencies
│   └── src/
│       ├── activation.rs    # Activation functions
│       ├── fully_connected.rs # Fully connected layers
│       ├── lib.rs           # Entry point for the library
│       ├── loss.rs          # Loss functions
│       ├── neural_network.rs # Neural network definition
│       ├── report.rs        # Reporting and result output
│       └── utils.rs         # Utility functions
└── src/                     # Main application for experiments
    ├── main.rs              # Entry point for the executable
    ├── mnist_experiment/    # MNIST-related modules
    │   ├── dataset_setup.rs # MNIST dataset preprocessing
    │   ├── mod.rs           # MNIST module entry point
    │   └── plot.rs          # Plotting results for MNIST
    └── rssi_experiment/     # RSSI-related modules
        ├── dataset_setup.rs # RSSI dataset preprocessing
        ├── mod.rs           # RSSI module entry point
        └── plot.rs          # Plotting results for RSSI
```

---

## 🛠️ Getting Started

### Prerequisites

- **Rust:** Install the latest version from [rust-lang.org](https://www.rust-lang.org/).

### Cloning the Repository

Clone the repository and navigate to the project folder:
```bash
git clone https://github.com/akaTsunemori/dnn-from-scratch.git
cd dnn-from-scratch
```

### Building the Project

To build the project, run: `cargo build --release`.

### Running Experiments
To run the experiments, run: `cargo run --release`.

---

## 🧪 Datasets

1. **MNIST Dataset:**
   - Stored in `assets/mnist/`.
   - Preprocessed as `.npy` files for seamless integration.

2. **RSSI Dataset:**
   - Found in `assets/rssi/rssi-dataset.csv`.
   - Contains signal strength data and coordinates (X, Y) for analysis.

---

## 📈 Results & Reporting

Each experiment generates reports and plots showcasing:
- Training history.
- Model performance metrics (e.g., accuracy for MNIST, CDF of RMSE for RSSI).

Plots and reports are stored in the in the output/ folder.

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

1. Fork the repo.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## 📧 Contact

For any inquiries or support, please create an issue.

---

Enjoy building your neural networks from scratch! 🎉
