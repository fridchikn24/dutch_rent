This project aims to predict rental prices in the Netherlands using a stacked ensemble model comprising k-Nearest Neighbors (k-NN), Light Gradient Boosting Machine (LightGBM), and Random Forest algorithms, with a final estimator of Random Forest.
Table of Contents

    Project Overview
    Installation
    Usage
    Model Architecture
    Contributing
    License

Project Overview

Accurately predicting rental prices is crucial for various stakeholders in the real estate market. This project leverages machine learning techniques to forecast rental prices based on available data. The ensemble approach combines the strengths of multiple algorithms to enhance prediction accuracy.
Installation

To set up the project locally, follow these steps:

    Clone the repository:

        git clone https://github.com/fridchikn24/dutch_rent.git

        cd dutch_rent

Install dependencies:

Ensure you have Poetry installed. Then, run:

    poetry install

    This command will create a virtual environment and install all required dependencies as specified in pyproject.toml.

Usage

    Activate the virtual environment:

poetry shell

Prepare the data:

Place your dataset (e.g., rent_apartments.csv) in the project directory. Ensure the data is preprocessed and cleaned as required by the model.

Train the model:

Execute the training script:

python src/train_model.py

This script will train the ensemble model and save the trained model to the specified directory.

Make predictions:

Use the trained model to make predictions on new data:

    python src/predict.py --input new_data.csv --output predictions.csv

    Replace new_data.csv with your input data file and predictions.csv with the desired output file name.

Model Architecture

The ensemble model consists of the following components:

    Base Learners:
        k-Nearest Neighbors (k-NN)
        Light Gradient Boosting Machine (LightGBM)
        Random Forest

    Final Estimator:
        Random Forest

The base learners are trained on the initial dataset, and their predictions are used as input features for the final estimator, which produces the final rental price predictions.
