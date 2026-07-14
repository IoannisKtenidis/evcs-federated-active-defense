import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

def initialize_pipeline():
    print("=== Starting OCPP FIDS Pipeline Initialization ===")
    
    # Define paths
    dataset_dir = "./dataset/OCPPFlow_meter"
    train_path = os.path.join(dataset_dir, "Train.csv")
    test_path = os.path.join(dataset_dir, "Test.csv")
    
    scaler_dir = "./scaler"
    encoder_dir = "./encoder"
    model_dir = "./model"
    
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    scaler_path = os.path.join(scaler_dir, "scaler_ocpp.joblib")
    encoder_path = os.path.join(encoder_dir, "encoder_ocpp.joblib")
    model_path = os.path.join(model_dir, "OCPP_model.h5")
    
    # 1. Load Dataset
    print(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path)
    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)
    
    # 2. Preprocess Data
    cols_to_drop = ['flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'flow_start_timestamp', 'flow_end_timestamp']
    
    df_train_cleaned = df_train.drop(columns=cols_to_drop, errors='ignore').dropna()
    df_test_cleaned = df_test.drop(columns=cols_to_drop, errors='ignore').dropna()
    
    X_train_raw = df_train_cleaned.drop(columns=['label'], errors='ignore')
    y_train_raw = df_train_cleaned['label']
    
    X_test_raw = df_test_cleaned.drop(columns=['label'], errors='ignore')
    y_test_raw = df_test_cleaned['label']
    
    print(f"Cleaned train shape: {X_train_raw.shape}, test shape: {X_test_raw.shape}")
    
    # 3. Fit and Save Label Encoder
    print("Fitting LabelEncoder...")
    encoder = LabelEncoder()
    # Fit on all labels to ensure all classes are captured
    all_labels = pd.concat([y_train_raw, y_test_raw])
    encoder.fit(all_labels)
    
    y_train_enc = encoder.transform(y_train_raw)
    y_test_enc = encoder.transform(y_test_raw)
    
    print("Class mapping:")
    for idx, class_name in enumerate(encoder.classes_):
        print(f"  {idx} -> {class_name}")
        
    joblib.dump(encoder, encoder_path)
    print(f"Saved LabelEncoder to {encoder_path}")
    
    # 4. Fit and Save StandardScaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    joblib.dump(scaler, scaler_path)
    print(f"Saved StandardScaler to {scaler_path}")
    
    # 5. Prepare targets for training (One-Hot Encoding)
    num_classes = len(encoder.classes_)
    y_train_onehot = tf.keras.utils.to_categorical(y_train_enc, num_classes=num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test_enc, num_classes=num_classes)
    
    # 6. Build and Train baseline MLP Model
    print("Building Keras MLP Model...")
    input_dim = X_train_scaled.shape[1]
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    print("Training model...")
    history = model.fit(
        X_train_scaled,
        y_train_onehot,
        validation_data=(X_test_scaled, y_test_onehot),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # 7. Evaluate Model
    print("Evaluating model...")
    y_pred_probs = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    test_acc = accuracy_score(y_test_enc, y_pred_classes)
    test_f1 = f1_score(y_test_enc, y_pred_classes, average='weighted')
    
    print(f"Baseline Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Baseline Test F1 Score: {test_f1:.4f}")
    
    # 8. Save Model
    model.save(model_path)
    print(f"Saved baseline model to {model_path}")
    print("=== Pipeline Initialization Completed Successfully ===")

if __name__ == "__main__":
    initialize_pipeline()
