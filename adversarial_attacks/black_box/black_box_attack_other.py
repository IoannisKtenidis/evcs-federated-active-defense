import json 
import os 
from white_box.utils import load_config
from white_box.attack_type import fgsm_attack, pgd_attack, bim_attack, jsma_attack, cw_attack
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import seaborn as sns

tf.disable_v2_behavior()

# Disable eager execution
tf.compat.v1.disable_eager_execution()

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


def preprocess(df_test, columns_to_drop):

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

    dim = X_scaled_test.shape[1]

    return X_scaled_test, y_test, scaler, dim

def main(config): 
    #config_path = 'config.json'

    #load the configuration
    #config = load_config(config_path) 

    #Access configuratuon parameters 
    dataset_path = config.get("dataset_path")
    dataset_name = config.get("dataset_name")
    #scaler_path = config.get("scaler_path")
    #model_path = config.get("model_path")
    eps = config.get("eps")
    evaluation_output_path = config.get("evaluation_output_path")
    adversarial_dataset_output_path = config.get("adversarial_dataset_output_path")
    columns_to_drop = config.get("columns_to_drop")
    type_of_attack = config.get("type_of_attack")
    jsma_theta = config.get("jsma_theta", 1.0)
    jsma_gamma = config.get("jsma_gamma", 0.1)

    #Print the configuration parameters to verify
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"Epsilon (eps): {eps}")
    print(f"Evaluation output path: {evaluation_output_path}")
    print(f"Adversarial dataset output path: {adversarial_dataset_output_path}")
    print(f"Columns to drop: {columns_to_drop}")
    print(f"Type of attack chosen: {type_of_attack}")
    print(f"JSMA theta: {jsma_theta}")
    print(f"JSMA gamma: {jsma_gamma}")
    
    #load dataset 
    df_test = pd.read_csv(dataset_path)

    # Preprocess the data
    X_scaled_test, y_test, scaler, dim = preprocess(df_test, columns_to_drop)
    print("Data preprocessed successfully.")

    model = Sequential([
        Dense(32, activation='relu', input_dim=dim),
        Dense(64, activation='tanh'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(y_test.value_counts()), activation='softmax'),  
    ]) 
    model.summary()

    #Define classifier
    from art.estimators.classification import KerasClassifier
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
        #elif attack == "deepfool":
            #X_test_adv = deepfool_attack(classifier, X_scaled_test)
        elif attack == "carlini":
            X_test_adv = cw_attack(classifier, X_scaled_test)    
        elif attack == "jsma":
            X_test_adv = jsma_attack(classifier, X_scaled_test, theta=jsma_theta, gamma=jsma_gamma)
        else:
            print(f"Unknown attack type: {attack}")
            continue

    # Denormalize the adversarial examples
    X_test_adv_denorm = scaler.inverse_transform(X_test_adv)

    X_test_adv_denorm_df = pd.DataFrame(X_test_adv_denorm, columns=X_scaled_test.columns)

    # Reattach dropped columns
    dropped_columns = df_test[columns_to_drop].iloc[X_test_adv_denorm_df.index]
    df_adv_complete = pd.concat([dropped_columns.reset_index(drop=True), X_test_adv_denorm_df.reset_index(drop=True)], axis=1)
    df_adv_complete['Label'] = y_test.values

    # Save the complete adversarial dataset
    denormalized_adversarial_dataset_path = os.path.join(adversarial_dataset_output_path, f'adversarial_dataset_{dataset_name}.csv')
    df_adv_complete.to_csv(denormalized_adversarial_dataset_path, index=False)
    print(f"Denormalized adversarial dataset saved to {denormalized_adversarial_dataset_path}")



    # Evaluate on Adversarial Test Data
    y_pred_adv = model.predict(X_test_adv)
    y_pred_adv_classes = np.argmax(y_pred_adv, axis=1)
    evaluation_metrics(y_test, y_pred_adv_classes, y_pred_adv, "Adversarial_Data", evaluation_output_path)


if __name__ == "__main__":
    config_path = 'black_box/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    main(config)
