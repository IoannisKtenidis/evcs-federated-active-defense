import os
import time
import csv
import datetime
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import joblib
from flask import Flask, request, jsonify
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Scalar
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod

# Disable warnings from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

SERVER_URL = os.environ.get("SERVER_URL", "http://fids-server:5000")
NODE_ID = os.environ.get("NODE_ID", "node_default")
NODE_URL = f"http://{NODE_ID}:5001"

# Local state
current_model_version = 1
model_status = "Weak (Initial)"
model_lock = threading.Lock()

class FedProxModel(tf.keras.Model):
    """
    Keras Model subclass to inject the FedProx proximal penalty
    into the loss function during local training.
    """
    def __init__(self, model: tf.keras.Model, proximal_mu: float = 0.0):
        super().__init__()
        self.inner_model = model
        self.proximal_mu = proximal_mu
        self.global_weights: List[tf.Tensor] = []

    def set_global_weights(self, global_weights: List[np.ndarray]):
        self.global_weights = [tf.constant(w, dtype=tf.float32) for w in global_weights]

    def call(self, inputs, training=None, mask=None):
        return self.inner_model(inputs, training=training)

    @property
    def metrics(self):
        return self.inner_model.metrics

    def train_step(self, data) -> Dict[str, tf.Tensor]:
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.inner_model(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
            # FedProx Proximal Regularization Penalty
            if self.proximal_mu > 0.0 and self.global_weights:
                proximal_penalty = 0.0
                for local_w, global_w in zip(self.inner_model.trainable_variables, self.global_weights):
                    proximal_penalty += tf.reduce_sum(tf.square(local_w - global_w))
                loss += (self.proximal_mu / 2.0) * proximal_penalty
                
        gradients = tape.gradient(loss, self.inner_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inner_model.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data) -> Dict[str, tf.Tensor]:
        x, y = data
        y_pred = self.inner_model(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


class FIDSClient(fl.client.NumPyClient):
    """Flower Client that runs robust adversarial training (FGSM-AT) locally."""
    def __init__(self, model: FedProxModel, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config) -> List[np.ndarray]:
        return self.model.inner_model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict]:
        print(f"[{NODE_ID}] Fit round started...")
        self.model.inner_model.set_weights(parameters)
        self.model.set_global_weights(parameters)
        
        # Read parameters from server config
        mu = float(config.get("proximal-mu", 0.01))
        self.model.proximal_mu = mu
        epochs = int(config.get("epochs", 5))
        batch_size = int(config.get("batch_size", 32))
        
        # Wrap Keras model with TensorFlowV2Classifier for ART evasion attack compatibility
        classifier = TensorFlowV2Classifier(
            model=self.model.inner_model,
            nb_classes=self.y_train.shape[-1],
            input_shape=self.x_train.shape[1:],
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            clip_values=(0.0, 1.0)
        )
        
        # Generate adversarial examples locally using FGSM (eps=0.1)
        attacker = FastGradientMethod(estimator=classifier, eps=0.1)
        x_train_adv = attacker.generate(x=self.x_train)
        
        # Mix clean and adversarial data (50-50 Robust Training Ratio)
        x_robust = np.concatenate([self.x_train, x_train_adv], axis=0)
        y_robust = np.concatenate([self.y_train, self.y_train], axis=0)
        
        # Shuffle robust dataset
        indices = np.arange(len(x_robust))
        np.random.shuffle(indices)
        x_robust = x_robust[indices]
        y_robust = y_robust[indices]
        
        # Fit model
        history = self.model.fit(
            x_robust,
            y_robust,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Save robust model locally
        robust_model_path = f"/app/robust_model_{NODE_ID}.h5"
        self.model.inner_model.save(robust_model_path)
        print(f"[{NODE_ID}] Saved local robust model to {robust_model_path}")
        
        # Update version state
        global current_model_version, model_status
        with model_lock:
            if current_model_version < 2:
                current_model_version = 2
                model_status = "Robust"
                print(f"[{NODE_ID}] Model state updated to Robust v2!")
                
        final_loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
        return self.model.inner_model.get_weights(), len(self.x_train), {"loss": final_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.model.inner_model.set_weights(parameters)
        loss, accuracy = self.model.inner_model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


def register_with_server():
    """Register once with the FIDS Server to get initialized."""
    global current_model_version, model_status
    while True:
        print(f"[{NODE_ID}] Attempting to register with FIDS Server at {SERVER_URL}...")
        try:
            response = requests.post(f"{SERVER_URL}/register", json={"node_id": NODE_ID, "node_url": NODE_URL})
            if response.status_code == 200:
                data = response.json().get('global_model', {})
                current_model_version = data.get("version", 1)
                model_status = data.get("status", "Weak (Initial)")
                print(f"[{NODE_ID}] Registered successfully! Global Model: v{current_model_version} ({model_status})")
                break
        except Exception as e:
            print(f"[{NODE_ID}] Registration failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def load_local_partition() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the client's local OCPP partitioned train and test datasets."""
    client_sub = "Client_1" if "node-1" in NODE_ID else "Client_2" if "node-2" in NODE_ID else "Client_3" if "node-3" in NODE_ID else "Client_1"
    train_path = f"/shared_aisap/dataset/OCPPFlow_meter/{client_sub}/Train.csv"
    test_path = f"/shared_aisap/dataset/OCPPFlow_meter/{client_sub}/Test.csv"
    
    # Load raw csv files
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Drop metadata columns
    cols_to_drop = ['flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'flow_start_timestamp', 'flow_end_timestamp']
    df_train = df_train.drop(columns=cols_to_drop, errors='ignore').dropna()
    df_test = df_test.drop(columns=cols_to_drop, errors='ignore').dropna()
    
    # Extract features and targets
    X_train_raw = df_train.drop(columns=['label'], errors='ignore')
    y_train_raw = df_train['label']
    X_test_raw = df_test.drop(columns=['label'], errors='ignore')
    y_test_raw = df_test['label']
    
    # Encode targets (multiclass to integers)
    encoder = joblib.load("/shared_aisap/encoder/encoder_ocpp.joblib")
    y_train_enc = encoder.transform(y_train_raw)
    y_test_enc = encoder.transform(y_test_raw)
    
    # One-hot encode targets
    y_train = tf.keras.utils.to_categorical(y_train_enc, num_classes=5)
    y_test = tf.keras.utils.to_categorical(y_test_enc, num_classes=5)
    
    # Scale features
    scaler = joblib.load("/shared_aisap/scaler/scaler_ocpp.joblib")
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    return X_train, y_train, X_test, y_test

def run_fl_client(server_address: str, strategy: str, mu: float):
    """Load local data and model, initialize client class, and run Flower client with connection retries."""
    print(f"[{NODE_ID}] Launching FL client connecting to {server_address}")
    
    try:
        X_train, y_train, X_test, y_test = load_local_partition()
        print(f"[{NODE_ID}] Local data loaded. Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Load baseline OCPP model
        model_path = "/shared_aisap/model/OCPP_model.h5"
        base_model = tf.keras.models.load_model(model_path)
        base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        # Initialize FedProx wrapped model
        fedprox_model = FedProxModel(model=base_model, proximal_mu=mu)
        fedprox_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        client = FIDSClient(model=fedprox_model, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
        
        # Retry connection loop to tolerate Flower server startup delays
        max_retries = 6
        retry_delay = 3
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[{NODE_ID}] Connecting to Flower server at {server_address} (attempt {attempt}/{max_retries})...")
                fl.client.start_client(server_address=server_address, client=client.to_client())
                print(f"[{NODE_ID}] FL client training round finished.")
                break
            except Exception as conn_err:
                print(f"[{NODE_ID}] Connection attempt {attempt} failed: {conn_err}")
                if attempt == max_retries:
                    print(f"[{NODE_ID}] Max connection retries reached. Exiting.")
                    raise conn_err
                time.sleep(retry_delay)
        
    except Exception as e:
        print(f"[{NODE_ID}] FL client execution crashed: {e}")

@app.route('/start_training', methods=['POST'])
def start_training():
    """Triggered by FIDS Server to participate in a Flower FL round."""
    data = request.get_json(force=True, silent=True) or {}
    server_address = data.get("server_address", "fids-server:8080")
    strategy = data.get("strategy", "fedprox")
    mu = float(data.get("mu", 0.01))
    
    # Start client in a daemon thread
    threading.Thread(target=run_fl_client, args=(server_address, strategy, mu), daemon=True).start()
    return jsonify({"status": "acknowledgement", "message": f"Client joining Flower round on {server_address}"}), 200

@app.route('/simulate_attack', methods=['POST'])
def simulate_attack():
    """Evaluate current model (v1 or v2) against real adversarial samples and alert n8n."""
    global current_model_version, model_status
    data = request.get_json(force=True, silent=True) or {}
    attack_payload = data.get("payload", "Unknown Attack")
    attack_type = data.get("attack_type", "fgsm")
    print(f"[{NODE_ID}] Received simulate_attack call (Type: {attack_type.upper()})")

    try:
        # 1. Select and load model
        model_version = current_model_version
        if model_version >= 2:
            model_path = f"/app/robust_model_{NODE_ID}.h5"
            print(f"[{NODE_ID}] Loading local ROBUST model (v{model_version}) from {model_path}")
        else:
            model_path = "/shared_aisap/model/OCPP_model.h5"
            print(f"[{NODE_ID}] Loading original WEAK model (v1) from {model_path}")
            
        model = tf.keras.models.load_model(model_path)
        
        # 2. Load and preprocess evaluation data
        adv_dataset_path = "/shared_aisap/results/adversarial_dataset.csv"
        
        if os.path.exists(adv_dataset_path):
            print(f"[{NODE_ID}] Loading generated adversarial dataset from {adv_dataset_path}...")
            df_adv = pd.read_csv(adv_dataset_path)
            
            # Drop metadata columns
            cols_to_drop = ['flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'flow_start_timestamp', 'flow_end_timestamp']
            df_adv = df_adv.drop(columns=cols_to_drop, errors='ignore').dropna()
            
            X_adv_raw = df_adv.drop(columns=['label'], errors='ignore')
            y_adv_raw = df_adv['label']
            
            # Encode labels and scale
            if pd.api.types.is_numeric_dtype(y_adv_raw):
                y_adv_classes = y_adv_raw.values.astype(int)
            else:
                encoder = joblib.load("/shared_aisap/encoder/encoder_ocpp.joblib")
                y_adv_classes = encoder.transform(y_adv_raw)
            y_adv_onehot = tf.keras.utils.to_categorical(y_adv_classes, num_classes=5)
            
            scaler = joblib.load("/shared_aisap/scaler/scaler_ocpp.joblib")
            X_adv = scaler.transform(X_adv_raw)
        else:
            # Fallback: Load local test set and generate adversarial examples on the fly using ART
            print(f"[{NODE_ID}] WARNING: Adversarial dataset file not found. Generating on-the-fly local evasion samples...")
            client_sub = "Client_1" if "node-1" in NODE_ID else "Client_2" if "node-2" in NODE_ID else "Client_3" if "node-3" in NODE_ID else "Client_1"
            test_path = f"/shared_aisap/dataset/OCPPFlow_meter/{client_sub}/Test.csv"
            
            df_test = pd.read_csv(test_path)
            cols_to_drop = ['flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'flow_start_timestamp', 'flow_end_timestamp']
            df_test = df_test.drop(columns=cols_to_drop, errors='ignore').dropna()
            
            X_test_raw = df_test.drop(columns=['label'], errors='ignore')
            y_test_raw = df_test['label']
            
            encoder = joblib.load("/shared_aisap/encoder/encoder_ocpp.joblib")
            y_adv_classes = encoder.transform(y_test_raw)
            
            scaler = joblib.load("/shared_aisap/scaler/scaler_ocpp.joblib")
            X_test_scaled = scaler.transform(X_test_raw)
            
            classifier = TensorFlowV2Classifier(
                model=model,
                nb_classes=5,
                input_shape=(49,),
                loss_object=tf.keras.losses.CategoricalCrossentropy(),
                clip_values=(0.0, 1.0)
            )
            attacker = FastGradientMethod(estimator=classifier, eps=0.1)
            X_adv = attacker.generate(x=X_test_scaled)
            
        # 3. Perform model evaluation on adversarial samples
        y_pred_prob = model.predict(X_adv, verbose=0)
        y_pred_classes = np.argmax(y_pred_prob, axis=1)
        
        acc = np.mean(y_pred_classes == y_adv_classes) * 100
        f1 = f1_score(y_adv_classes, y_pred_classes, average='weighted', zero_division=0)
        
        print(f"[{NODE_ID}] Real Evaluation Metrics -> Accuracy: {acc:.2f}%, F1-Score: {f1:.2f}")
        
        # 4. Save results to local attack history csv
        history_file = "/shared_aisap/results/attack_history.csv"
        is_new_file = not os.path.exists(history_file)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_str = "Compromised" if model_version < 2 else "Blocked"
        
        try:
            with open(history_file, "a", newline='') as f:
                writer = csv.writer(f)
                if is_new_file:
                    writer.writerow(["Timestamp", "Node ID", "Attack Type", "Status", "Model Version", "Accuracy", "F1 Score"])
                writer.writerow([timestamp, NODE_ID, attack_type.upper(), status_str, f"v{model_version}", f"{acc:.2f}%", f"{f1:.2f}"])
        except Exception as e:
            print(f"[{NODE_ID}] Error logging to attack history: {e}")
            
        # 5. Alert n8n orchestrator via webhook
        webhook_url = "http://n8n:5678/webhook/attack-alert"
        payload_to_n8n = {
            "node_id": NODE_ID,
            "attack_payload": attack_payload,
            "attack_type": attack_type,
            "model_version_used": model_version,
            "accuracy": f"{acc:.2f}%",
            "f1_score": f"{f1:.2f}",
            "status": "compromised" if model_version < 2 else "blocked",
            "result": "Το Σύστημα Παραβιάστηκε" if model_version < 2 else "Επιτυχής Απόκρουση Επίθεσης",
            "reason": f"Το μοντέλο είναι ευάλωτο! (Accuracy: {acc:.2f}%, F1: {f1:.2f})" if model_version < 2 else f"Ενεργοποιήθηκε η Ομόσπονδη Εκπαίδευση! (Accuracy: {acc:.2f}%, F1: {f1:.2f})"
        }
        
        try:
            requests.post(webhook_url, json=payload_to_n8n, timeout=5)
            print(f"[{NODE_ID}] Webhook alert successfully sent to n8n.")
        except Exception as e:
            print(f"[{NODE_ID}] Failed to send webhook alert to n8n: {e}")
            
        status_code = 400 if model_version < 2 else 200
        return jsonify(payload_to_n8n), status_code

    except Exception as e:
        print(f"[{NODE_ID}] Attack simulation failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Start registration thread
    threading.Thread(target=register_with_server, daemon=True).start()
    print(f"[{NODE_ID}] FIDS client Flask API starting on port 5001...")
    app.run(host='0.0.0.0', port=5001)
