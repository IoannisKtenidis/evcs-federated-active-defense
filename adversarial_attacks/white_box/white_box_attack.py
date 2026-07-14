import json 
import os 
from utils import load_config
from attack_type import (
    fgsm_attack, 
    pgd_attack, 
    bim_attack, 
    jsma_attack, 
    cw_attack,
    deepfool_attack,
    newtonfool_attack,
    elasticnet_attack,
    hopskipjump_attack
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
# tf.disable_v2_behavior()  # REMOVED to allow eager execution
# tf.compat.v1.disable_eager_execution() # REMOVED to allow eager execution

def evaluation_metrics(y_true, classes, predicted_test, dataset_name, evaluation_output_path):
    metrics = {}
    
    # Confusion Matrix
    cnf_matrix = confusion_matrix(y_true, classes)
    metrics['confusion_matrix'] = cnf_matrix.tolist()  # convert to list for JSON serialization
    print("Confusion Matrix:\n", cnf_matrix)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    confusion_matrix_path = os.path.join(evaluation_output_path, f'Confusion_Matrix_{dataset_name}.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Accuracy
    accuracy = accuracy_score(y_true, classes)
    metrics['accuracy'] = accuracy
    print('Accuracy: %f' % accuracy)

    # True positive rate - TPR
    TPR = TP / (TP + FN)
    metrics['TPR'] = np.mean(TPR)
    print("TPR: ", np.mean(TPR))

    # False positive rate - FPR
    FPR = FP / (FP + TN)
    metrics['FPR'] = np.mean(FPR)
    print("FPR: ", np.mean(FPR))

    # F1 Score
    f1 = f1_score(y_true, classes, average='weighted')
    metrics['f1_score'] = f1
    print('F1 score: %f' % f1)
    
    # AUC Score
    auc = roc_auc_score(y_true, predicted_test, multi_class='ovr')
    metrics['auc_score'] = auc
    print("AUC Score: %f" % auc)
    
    # Save metrics to JSON
    metrics_json_path = os.path.join(evaluation_output_path, f'Evaluation_Metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics  # Add this return statement


def preprocess(df_test, scaler_path, columns_to_drop, encoder=None):
    df_test_reduced = df_test.drop(columns_to_drop, axis=1, errors='ignore')
    df_test_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test_no_na = df_test_reduced.dropna(axis=0)

    # Identify the correct label column name (case-insensitive)
    label_col = next((col for col in df_test_no_na.columns if col.lower() == 'label'), None)
    if label_col is None:
        raise ValueError("No column named 'label' (case-insensitive) found in the dataframe.")

    y_test = df_test_no_na[label_col]
    X_test = df_test_no_na.drop([label_col], axis=1)

    if encoder:
        # Use provided encoder
        y_test = encoder.transform(y_test.values)
    else:
        # Fallback (not recommended for test set)
        le_y = LabelEncoder()
        y_test = le_y.fit_transform(y_test.values)
        
    y_test = pd.DataFrame(y_test, columns=['label_encoded'])

    # Load scaler and scale features
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    
    # Check for feature mismatch
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = set(scaler.feature_names_in_)
        current_features = set(X_test.columns)
        
        # Identifying extra columns in X_test that are not in the scaler
        extra_features = current_features - expected_features
        if extra_features:
            print(f"⚠️ Dropping {len(extra_features)} extra features not seen during training: {extra_features}")
            X_test = X_test.drop(columns=list(extra_features))

        # Reorder columns to match scaler's expected order
        X_test = X_test[scaler.feature_names_in_]

    X_scaled_test = scaler.transform(X_test)
    X_scaled_test = pd.DataFrame(X_scaled_test, columns=X_test.columns)

    return X_scaled_test, y_test

def merge_datasets(clear_df, denormalized_adversarial_dataset_path, dataset_name, encoder_path, adversarial_dataset_output_path):
    # Load the adversarial dataset
    df_adv_complete = pd.read_csv(denormalized_adversarial_dataset_path)

    # Load the pre-trained encoder
    enc = joblib.load(encoder_path)

    # Print the encoder classes (optional for debugging purposes)
    print(enc.classes_)

    # Assuming 'Label' is the column with encoded labels
    encoded_column_name = 'label'
    encoded_data = df_adv_complete[encoded_column_name]

    # Apply inverse_transform to the column and update 'Label' with the decoded values
    df_adv_complete['label'] = enc.inverse_transform(encoded_data)

    # Define the path for the full dataset with the decoded 'Label' column
    full_data_path = os.path.join(adversarial_dataset_output_path, f'decoded_{dataset_name}.csv')
    df_adv_complete.to_csv(full_data_path, index=False)

    # Print a message indicating that the dataset has been saved
    print(f"Full dataset with decoded labels saved to {full_data_path}")

    # Merge the two datasets (assuming they have the same columns)
    merged_df = pd.concat([clear_df, df_adv_complete], ignore_index=True)

    # Shuffle the merged dataframe
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)

    # Define path for the merged dataset
    merged_data_path = os.path.join(adversarial_dataset_output_path, f'merged_{dataset_name}.csv')
    
    # Save the shuffled dataframe to a new CSV file
    shuffled_df.to_csv(merged_data_path, index=False)

    print(f"Merged and shuffled dataset saved to {merged_data_path}")

def main(): 
    import os
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    # Load the configuration
    config = load_config(config_path)

    # Access configuration parameters 
    dataset_path = config.get("dataset_path")
    dataset_name = config.get("dataset_name")
    scaler_path = config.get("scaler_path")
    model_path = config.get("model_path")
    encoder_path = config.get("encoder_path")
    eps = config.get("eps")
    evaluation_output_path = config.get("evaluation_output_path")
    adversarial_dataset_output_path = config.get("adversarial_dataset_output_path")
    columns_to_drop = config.get("columns_to_drop")
    type_of_attack = config.get("type_of_attack")
    jsma_theta = config.get("jsma_theta", 1.0)
    jsma_gamma = config.get("jsma_gamma", 0.1)

    # Print the configuration parameters to verify
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"Scaler path: {scaler_path}")
    print(f"Model path: {model_path}")
    print(f'Encoder_path: {encoder_path}')
    print(f"Epsilon (eps): {eps}")
    print(f"Output path: {evaluation_output_path}")
    print(f"Adversarial dataset output path: {adversarial_dataset_output_path}")
    print(f"Columns to drop: {columns_to_drop}")
    print(f"Type of attack chosen: {type_of_attack}")
    print(f"JSMA theta: {jsma_theta}")
    print(f"JSMA gamma: {jsma_gamma}")

    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    model = load_model(model_path)
    model.summary()

    print(f"Scaler loaded from: {scaler_path}")
    print(f"Model loaded from: {model_path}")

    # Load dataset 
    df_test = pd.read_csv(dataset_path)
    
    # Preprocess the data
    X_scaled_test, y_test = preprocess(df_test, scaler_path, columns_to_drop)
    print("Data preprocessed successfully.")

    # Define classifier
    from art.estimators.classification import KerasClassifier
    # Force numpy input when constructing/using the classifier to avoid "tf.data.Dataset" iteration issues in TF1 mode
    scaled_columns = None
    if isinstance(X_scaled_test, pd.DataFrame):
        scaled_columns = X_scaled_test.columns
        X_scaled_test = X_scaled_test.values

    classifier = KerasClassifier(model=model, clip_values=(0,1))

    # Generate adversarial examples based on the type_of_attack
    for attack in type_of_attack:
        print(f"Running attack: {attack}")
        if attack == "fgsm":
            X_test_adv = fgsm_attack(classifier, X_scaled_test, eps)
        elif attack == "pgd":
            X_test_adv = pgd_attack(classifier, X_scaled_test, eps)
        elif attack == "bim":
            X_test_adv = bim_attack(classifier, X_scaled_test, eps)
        elif attack == "carlini":
            X_test_adv = cw_attack(classifier, X_scaled_test)
        elif attack == "jsma":
            X_test_adv = jsma_attack(classifier, X_scaled_test, theta=jsma_theta, gamma=jsma_gamma)
        else:
            print(f"Unknown attack type: {attack}")
            continue

    # Denormalize the adversarial examples
    X_test_adv_denorm = scaler.inverse_transform(X_test_adv)
    
    # If scaled_columns were saved, use them, otherwise use range
    if scaled_columns is not None:
        X_test_adv_denorm_df = pd.DataFrame(X_test_adv_denorm, columns=scaled_columns)
    else:
        X_test_adv_denorm_df = pd.DataFrame(X_test_adv_denorm)

    # Reattach dropped columns safely
    actual_columns = [col for col in columns_to_drop if col in df_test.columns]
    dropped_columns = df_test[actual_columns].iloc[X_test_adv_denorm_df.index]
    df_adv_complete = pd.concat([dropped_columns.reset_index(drop=True), X_test_adv_denorm_df.reset_index(drop=True)], axis=1)
    df_adv_complete['label'] = y_test.values

    # Save the complete adversarial dataset
    denormalized_adversarial_dataset_path = os.path.join(adversarial_dataset_output_path, f'adversarial_dataset_{dataset_name}.csv')
    df_adv_complete.to_csv(denormalized_adversarial_dataset_path, index=False)
    print(f"Denormalized adversarial dataset saved to {denormalized_adversarial_dataset_path}")

    # Evaluate on Adversarial Test Data
    y_pred_adv = model.predict(X_test_adv)
    y_pred_adv_classes = np.argmax(y_pred_adv, axis=1)
    evaluation_metrics(y_test, y_pred_adv_classes, y_pred_adv, f"Adversarial_Data_Evaluation", evaluation_output_path)

    # Auto-merge is disabled for Streamlit compatibility.
    # If merging is required, use api_server.py logic or add a Streamlit button.
    print("Exiting without merging to prevent EOFError in Streamlit.")

if __name__ == "__main__":
    main()