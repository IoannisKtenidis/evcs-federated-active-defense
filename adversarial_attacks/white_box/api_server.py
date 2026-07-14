from flask import Flask, request, jsonify, send_file
import os
import joblib
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from white_box_attack import preprocess, evaluation_metrics, fgsm_attack, pgd_attack, bim_attack, jsma_attack, cw_attack
from art.estimators.classification import KerasClassifier
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib

# Initialize Flask app
app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Welcome message for the API."""
    return jsonify({
        "message": "Welcome to the Adversarial Attack API. Use the provided endpoints to execute attacks.",
        "endpoints": {
            "POST /run_attack": "Run an adversarial attack by uploading files and specifying parameters.",
            "GET /download/<filename>": "Download the generated adversarial dataset."
        }
    })

@app.route('/run_attack', methods=['POST'])
def run_attack():
    """Handle adversarial attack execution."""
    try:
        # Retrieve uploaded files
        dataset_file = request.files.get('dataset')
        model_file = request.files.get('model')
        scaler_file = request.files.get('scaler')
        encoder_file = request.files.get('encoder')

        # Retrieve parameters
        epsilon = request.form.get('epsilon')
        attack_type = request.form.get('attack')
        columns_to_drop_input = request.form.get('columns_to_drop', "")
        download_dir = app.config['OUTPUT_FOLDER']

        # Default mock files if none are provided (fallback to OCPP configuration)
        dataset_path = 'dataset/OCPPFlow_meter/Test.csv'
        model_path = 'model/OCPP_model.h5'
        scaler_path = 'scaler/scaler_ocpp.joblib'
        encoder_path = 'encoder/encoder_ocpp.joblib'

        # If files were uploaded, override defaults
        if dataset_file and model_file and scaler_file and encoder_file:
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset_file.filename))
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
            scaler_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(scaler_file.filename))
            encoder_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(encoder_file.filename))

            dataset_file.save(dataset_path)
            model_file.save(model_path)
            scaler_file.save(scaler_path)
            encoder_file.save(encoder_path)

        # Default columns to drop if not explicitly provided (OCPP Flow IP/Port/Timestamp metadata)
        if not columns_to_drop_input:
            columns_to_drop_input = "flow_id,src_ip,dst_ip,src_port,dst_port,timestamp"

        # Validate required parameters
        if not epsilon or not attack_type:
            return jsonify({"error": "Both epsilon and attack type are required."}), 400

        epsilon = float(epsilon)
        columns_to_drop = [col.strip() for col in columns_to_drop_input.split(',') if col.strip()]

        # Ensure download directory exists
        os.makedirs(download_dir, exist_ok=True)

        # Load files
        df_test = pd.read_csv(dataset_path)
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)
        encoder = None
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)

        # Preprocess data
        X_scaled_test, y_test = preprocess(df_test, scaler_path, columns_to_drop, encoder=encoder)

        # Keep columns list and convert DataFrame to numpy for classifier compatibility
        scaled_columns = X_scaled_test.columns
        X_scaled_test_np = X_scaled_test.values

        # Create classifier
        classifier = KerasClassifier(model=model, clip_values=(0, 1))

        # Run selected attack
        attack_functions = {
            'fgsm': fgsm_attack,
            'pgd': pgd_attack,
            'bim': bim_attack,
            'jsma': lambda clf, x, eps: jsma_attack(clf, x, theta=1.0, gamma=0.1),
            'cw': cw_attack
        }

        if attack_type not in attack_functions:
            return jsonify({"error": f"Invalid attack type: {attack_type}. Available types: {list(attack_functions.keys())}"}), 400

        X_test_adv = attack_functions[attack_type](classifier, X_scaled_test_np, epsilon)

        # Denormalize and save adversarial examples
        X_test_adv_denorm = scaler.inverse_transform(X_test_adv)
        df_adv_complete = pd.DataFrame(X_test_adv_denorm, columns=scaled_columns)
        df_adv_complete['label'] = y_test.values

        # Define paths for adversarial dataset and metrics
        adversarial_dataset_path = os.path.join(app.config['OUTPUT_FOLDER'], 'adversarial_dataset.csv')
        metrics_path = os.path.join(app.config['OUTPUT_FOLDER'], 'metrics.json')

        # Save adversarial examples to CSV
        df_adv_complete.to_csv(adversarial_dataset_path, index=False)

        # Verify file creation
        if not os.path.exists(adversarial_dataset_path):
            raise FileNotFoundError(f"Failed to save adversarial dataset to: {adversarial_dataset_path}")

        # Evaluate metrics and save to the same output folder
        y_pred_adv = model.predict(X_test_adv)
        y_pred_adv_classes = np.argmax(y_pred_adv, axis=1)
        y_test_1d = y_test['label_encoded'].values
        metrics = evaluation_metrics(y_test_1d, y_pred_adv_classes, y_pred_adv, 'Adversarial_Dataset', app.config['OUTPUT_FOLDER'])

        # Save metrics to a file in the output folder
        with open(metrics_path, 'w') as metrics_file:
            import json
            json.dump(metrics, metrics_file)

        # Return metrics and file paths to the client
        return jsonify({
            "message": "Attack completed successfully.",
            #"metrics": {k: v for k, v in metrics.items() if k != 'confusion_matrix'},  # Include metrics in the response
            #"metrics_path": metrics_path,
            "adversarial_dataset_path": "/download/adversarial_dataset.csv"
        })


    except Exception as e:
        # Return detailed error for debugging
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated adversarial dataset."""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": f"File not found: {file_path}"}), 404

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
