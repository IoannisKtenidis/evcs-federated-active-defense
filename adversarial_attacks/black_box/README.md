<p align="center">
  <img src="logo.png" alt="AAG Logo" width="200"/>
</p>

# Adversarial Attack Project

## Overview

This project implements various black-box adversarial attacks on machine learning models. The goal is to evaluate the robustness of these models against such attacks. The project includes implementations of different attack types, utilities for handling data, and configurations for running the attacks. It contains a black box attack (ZOO) which uses a open source model Random Forest and black box attack using different kind of evasion attacks (FGSM, JSMA, BIM, PGD, C$W) using a standard Sequential DNN model wich is trained each time to the dataset provided.

### Requirements
- Python 3.11.0
- adversarial_robustness-toolbox = 1.18.0
- numpy
- pandas
- scikit-learn
- keras
- tensorflow
- joblib
- matplotlib
- seaborn 

## Project Structure

- `main.py`: The main script to run the adversarial attacks.
- `black_box_attack_other.py`: Contains the implementation of evasion white box attacks using the Sequential model.
- `black_box_attacks.py`: Contains the implementation Black-box attacks (ZOO, HopSkipJump, Boundary, Square).
- `attack_type.py`: Defines the different types of attacks used in the project.
- `config.json`: Configuration file for specifying paths, attack parameters, and other settings.
- `utils.py`: Utility functions used throughout the project.

## Configuration

The `config.json` file contains the configuration settings for the project. Below is a brief description of each field:

Configuration file path: 'black_box\config.json'
- `dataset_path`: Path to the dataset file (CSV format).
- `eps`: Epsilon value used for the attacks.
- `evaluation_output_path`: Path where the evaluation results will be saved (JSON format).
- `adversarial_dataset_output_path`: Path where the adversarial dataset will be saved (CSV format).
- `columns_to_drop`: List of columns to be dropped from the dataset.
- `type_of_attack`: List of attack types to be used.
- `jsma_theta`: This parameter controls the amount of perturbation applied to each feature during the attack. (parameter if JSMA is used)
- `jsma_gamma`: This parameter determines the maximum proportion of features that can be perturbed in the input. (parameter if JSMA is used)

Example `config.json`:

```json
{
    "dataset_path": "black_box/dataset/test.csv",
    "dataset_name" : "blackbox_test",
    "eps": 0.1, 
    "evaluation_output_path": "black_box/results",
    "adversarial_dataset_output_path": "black_box/results",
    "columns_to_drop": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port","Protocol", "Timestamp"],
    "type_of_attack": ["fgsm"],
    "jsma_theta": 0.9,
    "jsma_gamma": 0.01
}

```

type_of_attacks -> (fgsm, pgd, jsma, carlini, bim, zoo, hopskipjump/hsj, square, nes)
columns_to_drop -> Add the columns that are not needed for training

## Usage

### Running the Attacks

To run the attacks, execute the `main.py` script:

```bash
python main.py
```

Ensure that the `config.json` file is correctly set up with the necessary paths and parameters before running the script.

### Implementing New Attacks

To implement a new attack:

1. Define the attack in a new script or within an existing script like `black_box_attack_other.py`.
2. Add the necessary configuration options in `config.json`.
3. Modify `main.py` to include the new attack type.

## Requirements

Make sure to install all required dependencies before running the project. You can install them using pip:

```bash
pip install -r requirements.txt
```


## Example 

There is an junpyter notebook file as an example of using ZOO attack 

```bash
jupyter blackbox_attack_example.ipynb
```

## Dashboard 

To test the black box attacks you can run the following command to open the dashboard

```bash
streamlit run app.py
```