# Transformers Explained

This repository contains code for training and evaluating deep learning models using PyTorch, with support for CNN, ResNet, and Vision Transformer architectures. It also includes TensorBoard integration for comprehensive logging and monitoring of training and validation metrics. The primary objective is to investigate how each of the different back bones contribute to feature extraction. 

## Table of Contents

- [Transformers Explained](#transformers-explained)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Training and Evaluation](#training-and-evaluation)
  - [TensorBoard](#tensorboard)
  - [UnitTests](#unittests)
  - [Usage Example](#usage-example)
  - [Acknowledgements](#acknowledgements)

## Project Structure

``` aurduino

transformers_explained/
├── config/
│   └── config.yaml
├── layers/
│   └── classification_head.py
├── models/
│   ├── cnn.py
│   ├── generic_model.py
│   ├── resnet.py
│   └── vit.py
├── train.py
├── utils/
│   ├── data_loader.py
│   └── visualizations.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_generic_model.py
│   ├── test_vit.py
│   └── __init__.py
├── requirements.txt
└── README.md


```

## Installation

To set up this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/transformers_explained.git
   cd transformers_explained 
   ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The configuration for the project is managed using Hydra and is stored in config/con.yaml. Here you can set various parameters such as the model type, training parameters, and logging settings.

## Training and Evaluation

To train and evaluate the model, run the following command:

```bash
python train.py
```

This script will:
    Train the model specified in the configuration file.
    Evaluate the model on the validation dataset at specified intervals.
    Log the training and validation metrics to TensorBoard.

## TensorBoard

TensorBoard is used for logging and visualizing training progress and evaluation metrics. To start TensorBoard, run:

```bash
tensorboard --logdir=./logs
```

Then open your web browser and go to http://localhost:6006/ to view the TensorBoard dashboard.

## UnitTests

Unit tests are provided to ensure the correctness of the data loader, models, and the overall training pipeline. To run the unit tests, use pytest:

```bash
python -m pytest tests/
```

## Usage Example

```bash
python train.py --config-name=conf.yaml
```

Make sure to adjust the configuration file (config/config.yaml) to your needs before running the script.

## Acknowledgements

This project structure and configuration are inspired by best practices for machine learning projects and the use of modern tools like Hydra and TensorBoard for configuration and logging.

Feel free to contribute to this project by opening issues or submitting pull requests.
