# black_box_attacks.py
import json
import os
import warnings
from typing import Dict, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from art.estimators.classification import SklearnClassifier

from white_box.attack_type import (
    hsj_attack,          # HopSkipJump
    square_attack,       # Square Attack
    boundary_attack,     # Boundary Attack
    zoo_attack           # ZOO
)

warnings.filterwarnings("ignore")


def preprocess(df_test: pd.DataFrame, columns_to_drop):
    """Standard tabular preprocessing: drop cols, drop NAs, encode label, scale features."""
    df_test_reduced = df_test.drop(columns_to_drop, axis=1)
    df_test_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test_no_na = df_test_reduced.dropna(axis=0)

    # Identify the correct label column name (case-insensitive)
    label_col = next((col for col in df_test_no_na.columns if col.lower() == 'label'), None)
    if label_col is None:
        raise ValueError("No column named 'label' (case-insensitive) found in the dataframe.")

    y_test = df_test_no_na[label_col]
    X_test = df_test_no_na.drop([label_col], axis=1)

    le_y = LabelEncoder()
    y_test = le_y.fit_transform(y_test.values)
    y_test = pd.DataFrame(y_test, columns=['label_encoded'])
    
    scaler = StandardScaler()
    X_scaled_test = scaler.fit_transform(X_test)
    X_scaled_test = pd.DataFrame(X_scaled_test, columns=X_test.columns)

    return X_scaled_test, y_test, scaler


def evaluation_metrics(y_true, classes, predicted_test, dataset_name, evaluation_output_path):
    """Compute and save confusion matrix, accuracy, TPR/FPR mean, and F1."""
    os.makedirs(evaluation_output_path, exist_ok=True)
    metrics = {}

    cnf_matrix = confusion_matrix(y_true, classes)
    metrics["confusion_matrix"] = cnf_matrix.tolist()
    print("Confusion Matrix:\n", cnf_matrix)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    confusion_matrix_path = os.path.join(
        evaluation_output_path, f"Confusion_Matrix_{dataset_name}.png"
    )
    plt.savefig(confusion_matrix_path, bbox_inches="tight")
    plt.close()

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    accuracy = accuracy_score(y_true, classes)
    metrics["accuracy"] = float(accuracy)
    print("Accuracy: %f" % accuracy)

    TPR = TP / (TP + FN + 1e-12)
    metrics["TPR"] = float(np.mean(TPR))
    print("TPR: ", np.mean(TPR))

    FPR = FP / (FP + TN + 1e-12)
    metrics["FPR"] = float(np.mean(FPR))
    print("FPR: ", np.mean(FPR))

    f1 = f1_score(y_true, classes, average="weighted")
    metrics["f1_score"] = float(f1)
    print("F1 score: %f" % f1)

    metrics_json_path = os.path.join(
        evaluation_output_path, f"Evaluation_Metrics_{dataset_name}.json"
    )
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)


def main(config):
    # --- Read config ---
    dataset_path = config.get("dataset_path")
    dataset_name = config.get("dataset_name", "dataset")
    evaluation_output_path = config.get("evaluation_output_path", "black_box/results")
    adversarial_dataset_output_path = config.get("adversarial_dataset_output_path", "black_box/results")
    columns_to_drop = config.get("columns_to_drop", [])
    type_of_attack = config.get("type_of_attack", ["zoo"])  # e.g., ["zoo", "hsj", "boundary", "square"]

    print(f"Dataset path: {dataset_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"Evaluation output path: {evaluation_output_path}")
    print(f"Adversarial dataset output path: {adversarial_dataset_output_path}")
    print(f"Columns to drop: {columns_to_drop}")
    print(f"Attacks to run: {type_of_attack}")

    # --- Load dataset ---
    df_test = pd.read_csv(dataset_path)

    # --- Preprocess ---
    X_scaled_test_df, y_test_df, scaler = preprocess(df_test, columns_to_drop)
    print("Data preprocessed successfully.")

    # Convert to numpy and flatten labels
    X_scaled_np = X_scaled_test_df.to_numpy()
    y_true = y_test_df.values.ravel()

    # --- Train a simple surrogate (RandomForest) and wrap for ART ---
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X=X_scaled_np, y=y_true)
    clf_rf = SklearnClassifier(model=rf)  # features are standardized; clip_values not required

    # Ensure output dirs exist
    os.makedirs(adversarial_dataset_output_path, exist_ok=True)
    os.makedirs(evaluation_output_path, exist_ok=True)

    # --- Attack dispatch (all functions come from attack_type.py) ---
    dispatch: Dict[str, Callable] = {
        "zoo": zoo_attack,
        "hopskipjump": hsj_attack,
        "hsj": hsj_attack,
        "hop-skip-jump": hsj_attack,
        "boundary": boundary_attack,
        "boundaryattack": boundary_attack,
        "ba": boundary_attack,
        "square": square_attack,
        "square_attack": square_attack,
        "squareattack": square_attack,
    }

    # --- Run each requested black-box attack ---
    for attack_name in type_of_attack:
        key = attack_name.lower()
        if key not in dispatch:
            print(f"[skip] '{attack_name}' is not a supported black-box attack in this module.")
            continue

        print(f"Running black-box attack: {attack_name}")
        attack_fn = dispatch[key]

        # Generate adversarial examples via the function defined in attack_type.py
        X_adv = attack_fn(clf_rf, X_scaled_np)

        # --- Denormalize back to original feature scale ---
        X_adv_denorm = scaler.inverse_transform(X_adv)

        # Rebuild a DataFrame with original (kept) columns
        kept_columns = df_test.drop(columns=columns_to_drop + ["Label"]).columns
        if len(kept_columns) != X_adv_denorm.shape[1]:
            raise ValueError("Column mismatch while reconstructing adversarial DataFrame.")
        X_adv_denorm_df = pd.DataFrame(X_adv_denorm, columns=kept_columns)

        # Reattach the dropped columns from the original rows (matching by index)
        dropped_cols_df = df_test[columns_to_drop].iloc[X_adv_denorm_df.index].reset_index(drop=True)
        df_adv_complete = pd.concat([dropped_cols_df, X_adv_denorm_df.reset_index(drop=True)], axis=1)

        # Add label column (still using encoded labels for consistency)
        df_adv_complete["Label"] = y_true

        # Save adversarial dataset per attack
        safe_name = key.replace("-", "")
        adv_csv_path = os.path.join(
            adversarial_dataset_output_path,
            f"denormalized_adversarial_dataset_{dataset_name}_{safe_name}.csv",
        )
        df_adv_complete.to_csv(adv_csv_path, index=False)
        print(f"[{attack_name}] Adversarial dataset saved to: {adv_csv_path}")

        # --- Evaluate surrogate model on adv examples ---
        y_pred_adv = rf.predict(X_adv)
        evaluation_metrics(
            y_true=y_true,
            classes=y_pred_adv,
            predicted_test=y_pred_adv,
            dataset_name=f"Adversarial_Data_{safe_name}",
            evaluation_output_path=evaluation_output_path,
        )


if __name__ == "__main__":
    config_path = "black_box/config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    main(cfg)
