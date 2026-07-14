import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def evaluate():
    clean_dataset_path = "/shared_aisap/dataset/OCPPFlow_meter/Test.csv"
    adv_dataset_path = "/shared_aisap/results/adversarial_dataset.csv"
    
    weak_model_path = "/shared_aisap/model/OCPP_model.h5"
    robust_model_path = f"/app/robust_model_{os.environ.get('NODE_ID', 'fids-node-1')}.h5"
    
    print("\n=======================================================")
    print("🛡️ AMYNA-TN Scenario 2: IN-DEPTH METRICS 🛡️")
    print("=======================================================\n")
    
    # 1. Load Adversarial Dataset
    if not os.path.exists(adv_dataset_path):
        print("❌ Error: Adversarial Dataset not found at results/adversarial_dataset.csv!")
        return
        
    cols_to_drop = ['flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'flow_start_timestamp', 'flow_end_timestamp']
    df_adv = df_adv.drop(columns=cols_to_drop, errors='ignore').dropna()
    
    y_adv_raw = df_adv['label']
    X_adv_raw = df_adv.drop(columns=['label'], errors='ignore').values
    
    # Scale Data
    try:
        scaler = joblib.load("/shared_aisap/scaler/scaler_ocpp.joblib")
        X_adv = scaler.transform(X_adv_raw)
    except Exception as e:
        print(f"⚠️ Failed to scale adversarial evaluation data: {e}. Using raw features...")
        X_adv = X_adv_raw
        
    # Encode targets
    if pd.api.types.is_numeric_dtype(y_adv_raw):
        y_adv = y_adv_raw.values.astype(int)
    else:
        try:
            encoder = joblib.load("/shared_aisap/encoder/encoder_ocpp.joblib")
            y_adv = encoder.transform(y_adv_raw)
        except Exception as e:
            print(f"❌ Failed to encode targets: {e}")
            return
    
    #########################################################
    # STEP 1: ORIGINAL MODEL (Weak)
    #########################################################
    print("--- STEP 1: Original Model (Pre-Attack) ---")
    if os.path.exists(weak_model_path):
        weak_model = tf.keras.models.load_model(weak_model_path)
        weak_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        y_pred_prob_adv1 = weak_model.predict(X_adv, verbose=0)
        y_pred_adv1 = np.argmax(y_pred_prob_adv1, axis=1)
        acc_adv1 = np.mean(y_pred_adv1 == y_adv)
        f1_adv1 = f1_score(y_adv, y_pred_adv1, average='weighted', zero_division=0)
        prec_adv1 = precision_score(y_adv, y_pred_adv1, average='weighted', zero_division=0)
        rec_adv1 = recall_score(y_adv, y_pred_adv1, average='weighted', zero_division=0)

        print(f"[Under Attack]   Accuracy : {acc_adv1*100:.2f}% ⬇️ (System Compromised)")
        print(f"Metrics (Attack) ->  F1-Score: {f1_adv1:.2f} | Precision: {prec_adv1:.2f} | Recall: {rec_adv1:.2f}")
    else:
        print("❌ Original Model file not found.")
        return

    #########################################################
    # STEP 2: FEDERATED LEARNING MODEL (Robust)
    #########################################################
    print("\n" + "-"*50 + "\n")
    print("--- STEP 2: Robust Model (Post-Federated Learning) ---")
    if os.path.exists(robust_model_path):
        robust_model = tf.keras.models.load_model(robust_model_path)
        robust_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        y_pred_prob_adv2 = robust_model.predict(X_adv, verbose=0)
        y_pred_adv2 = np.argmax(y_pred_prob_adv2, axis=1)
        
        acc_adv2 = np.mean(y_pred_adv2 == y_adv)
        f1_adv2 = f1_score(y_adv, y_pred_adv2, average='weighted', zero_division=0)
        prec_adv2 = precision_score(y_adv, y_pred_adv2, average='weighted', zero_division=0)
        rec_adv2 = recall_score(y_adv, y_pred_adv2, average='weighted', zero_division=0)

        print(f"[Defended]       Accuracy : {acc_adv2*100:.2f}% ⬆️ (Defense Active!)")
        print(f"Metrics (Defended)-> F1-Score: {f1_adv2:.2f} | Precision: {prec_adv2:.2f} | Recall: {rec_adv2:.2f}")
        
        print("\n📈 TOTAL IMPROVEMENT: ")
        print(f"   Accuracy:  +{((acc_adv2 - acc_adv1)*100):.2f}%")
        print(f"   F1-Score:  +{(f1_adv2 - f1_adv1):.2f}")
        
    else:
        print("⚠️ Robust Model file not found. Run the N8N Retraining Workflow first.")

    print("\n=======================================================\n")

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    evaluate()
