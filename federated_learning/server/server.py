import multiprocessing
import os
import signal
import socket
import time
import psutil
import requests
import threading
import numpy as np
from flask import Flask, request, jsonify, Response
from typing import Optional, Tuple, Dict, List
import flwr as fl
from flwr.common import Metrics, Scalar, ndarrays_to_parameters, parameters_to_ndarrays

# Define standard spawn method for multiprocessing to prevent TensorFlow locks
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

app = Flask(__name__)

# Global model info and client registry
GLOBAL_MODEL_INFO = {"version": 1, "status": "Weak (Initial)", "description": "Initial OCPP FIDS Model"}
NODES = {} # Dictionary mapping node_id -> node_url
flower_process: Optional[multiprocessing.Process] = None

def check_port_open(port: int) -> bool:
    """Check if a port is bound (in use)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def kill_process_by_port(port: int) -> None:
    """Identify and terminate any process listening on the specified port recursively."""
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            pid = conn.pid
            if pid is None or pid == 0:
                continue
            try:
                proc = psutil.Process(pid)
                print(f"[SERVER CLEANUP] Terminating process {proc.name()} (PID: {pid}) listening on port {port}")
                
                # Terminate children recursively
                for child in proc.children(recursive=True):
                    if child.status() != psutil.STATUS_ZOMBIE:
                        child.send_signal(signal.SIGKILL)
                
                # Terminate parent
                if proc.status() != psutil.STATUS_ZOMBIE:
                    proc.terminate()
                
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1)
                
                if proc.status() == psutil.STATUS_ZOMBIE:
                    print(f"[SERVER CLEANUP] Cleaned up zombie PID {pid}.")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"[SERVER WARNING] Failed to access process PID {pid}: {e}")

def save_global_model(ndarrays: List[np.ndarray]):
    """Update global model weights on disk with the aggregated weights from Flower."""
    import tensorflow as tf
    model_path = "/shared_aisap/model/OCPP_model.h5"
    if os.path.exists(model_path):
        try:
            # We disable GPU inside the aggregator server to save VRAM and avoid memory limits
            tf.config.set_visible_devices([], 'GPU')
            model = tf.keras.models.load_model(model_path)
            model.set_weights(ndarrays)
            model.save(model_path)
            print(f"[SERVER AGGREGATION] Robust aggregated weights successfully saved to {model_path}")
        except Exception as e:
            print(f"[SERVER ERROR] Failed to save aggregated weights: {e}")
    else:
        print(f"[SERVER ERROR] Model file not found at {model_path} during aggregation.")

# Define robust strategy subclasses to intercept aggregated weights
class CustomFedProx(fl.server.strategy.FedProx):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            save_global_model(parameters_to_ndarrays(aggregated_parameters))
        return aggregated_parameters, aggregated_metrics

class CustomFedMedian(fl.server.strategy.FedMedian):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            save_global_model(parameters_to_ndarrays(aggregated_parameters))
        return aggregated_parameters, aggregated_metrics

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            save_global_model(parameters_to_ndarrays(aggregated_parameters))
        return aggregated_parameters, aggregated_metrics

def fit_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate training fit metrics (losses) from clients."""
    if not metrics:
        return {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    weighted_loss = sum([num_examples * float(m.get("loss", 0.0)) for num_examples, m in metrics]) / total_examples
    return {"aggregated_loss": weighted_loss}

def evaluate_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate validation metrics (accuracies) from clients."""
    if not metrics:
        return {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    weighted_accuracy = sum([num_examples * float(m.get("accuracy", 0.0)) for num_examples, m in metrics]) / total_examples
    return {"aggregated_accuracy": weighted_accuracy}

def get_initial_parameters() -> fl.common.Parameters:
    """Load the initial model weights from disk to instantiate the global parameters."""
    import tensorflow as tf
    # Disable GPU to save memory
    tf.config.set_visible_devices([], 'GPU')
    model_path = "/shared_aisap/model/OCPP_model.h5"
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            weights = model.get_weights()
            print(f"[SERVER] Extracted initial parameters from Keras model: {model_path}")
            return ndarrays_to_parameters(weights)
        except Exception as e:
            print(f"[SERVER WARNING] Error reading Keras model: {e}. Generating dummy weights.")
    
    # Fallback to random weights if model is missing
    dummy_weights = [
        np.random.randn(49, 64).astype(np.float32) * 0.01,
        np.zeros(64, dtype=np.float32),
        np.random.randn(64, 32).astype(np.float32) * 0.01,
        np.zeros(32, dtype=np.float32),
        np.random.randn(32, 5).astype(np.float32) * 0.01,
        np.zeros(5, dtype=np.float32)
    ]
    return ndarrays_to_parameters(dummy_weights)

def run_flower_server(strategy_name: str, mu: float, rounds: int) -> None:
    """Isolated target function to execute the Flower FL Server loop."""
    print(f"[FL SERVER] Initializing Flower server (Strategy: {strategy_name}, Rounds: {rounds})...")
    
    initial_params = get_initial_parameters()
    
    if strategy_name == "fedprox":
        print(f"[FL SERVER] Strategy FedProx with mu={mu}")
        strategy = CustomFedProx(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            proximal_mu=mu,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        )
    elif strategy_name == "fedmedian":
        print("[FL SERVER] Strategy FedMedian")
        strategy = CustomFedMedian(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        )
    else:
        print("[FL SERVER] Strategy FedAvg")
        strategy = CustomFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        )
    
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy
        )
        print("[FL SERVER] Flower execution completed successfully.")
    except Exception as e:
        print(f"[FL SERVER ERROR] Execution failed: {e}")

@app.route('/register', methods=['POST'])
def register():
    """Register IoT clients at startup."""
    data = request.get_json(force=True, silent=True) or {}
    node_id = data.get("node_id")
    node_url = data.get("node_url", f"http://{node_id}:5001")
    if node_id not in NODES:
        NODES[node_id] = node_url
        print(f"[API SERVER] Registered node: {node_id} at {node_url}")
    return jsonify({"status": "success", "global_model": GLOBAL_MODEL_INFO}), 200

@app.route('/model', methods=['GET'])
def get_model():
    """Return info about the current global model state."""
    return jsonify(GLOBAL_MODEL_INFO), 200

@app.route('/retrain', methods=['POST'])
def retrain() -> Tuple[Response, int]:
    """Trigger retraining process, clearing port 8080 and starting a new Flower server process."""
    global flower_process, GLOBAL_MODEL_INFO
    print("[API SERVER] Received trigger from orchestrator to start retraining.")
    
    # 1. Terminate any currently active server process
    if flower_process is not None:
        if flower_process.is_alive():
            print(f"[API SERVER] Terminating active Flower server process (PID: {flower_process.pid})")
            flower_process.terminate()
            flower_process.join(timeout=3)
            if flower_process.is_alive():
                print("[API SERVER] Flower process unresponsive. Killing.")
                flower_process.kill()
                flower_process.join()
        flower_process = None
        
    # 2. Cleanup orphaned bindings on port 8080
    kill_process_by_port(8080)
    
    # 3. Verify port is freed up
    port_free = False
    for _ in range(5):
        if not check_port_open(8080):
            port_free = True
            break
        print("[API SERVER] Waiting for port 8080 to be released...")
        time.sleep(1)
        
    if not port_free:
        return jsonify({
            "status": "error",
            "message": "Port 8080 is blocked. Cannot start FL server."
        }), 409
        
    # Read parameters from n8n trigger payload
    data = request.get_json(force=True, silent=True) or {}
    strategy = data.get("strategy", "fedprox").lower()
    mu = float(data.get("mu", 0.01))
    rounds = int(data.get("rounds", 3))
    
    # 4. Spawn Flower Server process
    flower_process = multiprocessing.Process(
        target=run_flower_server, 
        args=(strategy, mu, rounds)
    )
    flower_process.start()
    print(f"[API SERVER] Flower server started with PID: {flower_process.pid}")
    
    # 5. Notify all registered clients asynchronously to start training and connect
    def notify_clients():
        time.sleep(2) # wait for Flower gRPC server to bind successfully
        for node_id, node_url in NODES.items():
            try:
                print(f"[API SERVER] Instructing client {node_id} to start training...")
                res = requests.post(
                    f"{node_url}/start_training", 
                    json={"server_address": "fids-server:8080", "strategy": strategy, "mu": mu}, 
                    timeout=5
                )
                print(f"[API SERVER] Response from {node_id}: {res.status_code}")
            except Exception as e:
                print(f"[API SERVER ERROR] Failed to notify client {node_id}: {e}")
                
    threading.Thread(target=notify_clients, daemon=True).start()
    
    # Update global model state info
    GLOBAL_MODEL_INFO["version"] += 1
    GLOBAL_MODEL_INFO["status"] = "Robust"
    GLOBAL_MODEL_INFO["description"] = f"Adversarially trained OCPP FIDS Model via {strategy.upper()}"
    
    return jsonify({
        "status": "success",
        "message": "Flower FL Retraining triggered.",
        "strategy": strategy,
        "rounds": rounds,
        "pid": flower_process.pid
    }), 200

@app.route('/status', methods=['GET'])
def get_status() -> Tuple[Response, int]:
    """Retrieve operational status of the server process."""
    global flower_process
    is_alive = flower_process is not None and flower_process.is_alive()
    port_active = check_port_open(8080)
    return jsonify({
        "flower_server_process_running": is_alive,
        "port_8080_in_use": port_active,
        "pid": flower_process.pid if is_alive else None,
        "registered_nodes": list(NODES.keys())
    }), 200

if __name__ == '__main__':
    print("[SERVER] FIDS Federated Aggregator listening on port 5000...")
    app.run(host='0.0.0.0', port=5000)
