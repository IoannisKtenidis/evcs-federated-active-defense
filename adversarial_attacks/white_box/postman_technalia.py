from flask import Flask, request, jsonify, send_file, make_response
import os
import joblib
import pandas as pd
import numpy as np
import uuid  # Generate unique attack IDs
import zipfile  # Package files into a single response
from io import BytesIO  # Handle files in memory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from white_box_attack import preprocess, evaluation_metrics, fgsm_attack, pgd_attack, bim_attack, jsma_attack, cw_attack
from art.estimators.classification import KerasClassifier
import matplotlib
import json

# ✅ Suppress TensorFlow and Scikit-learn warnings for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Reduce log verbosity

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Welcome message for the API."""
    return jsonify({
        "message": "Welcome to the Adversarial Attack API. Use the provided endpoints to execute attacks.",
        "endpoints": {
            "POST /run_attack": "Run an adversarial attack and receive the dataset and metrics in a single request."
        }
    })

@app.route('/run_attack', methods=['POST'])
def run_attack():
    """Handle adversarial attack execution and return dataset + metrics in a single response."""
    try:
        # Generate a unique attack ID
        attack_id = str(uuid.uuid4())[:8]  # Shorter unique ID for easier tracking

        # Retrieve uploaded files
        dataset_file = request.files.get('dataset')
        model_file = request.files.get('model')
        scaler_file = request.files.get('scaler')
        encoder_file = request.files.get('encoder')

        # Retrieve parameters
        epsilon = request.form.get('epsilon')
        attack_type = request.form.get('attack')  # Capture the attack type
        columns_to_drop_input = request.form.get('columns_to_drop', "")

        # Validate required files
        if not all([dataset_file, model_file, scaler_file, encoder_file]):
            return jsonify({"error": "All files (dataset, model, scaler, encoder) are required."}), 400

        # Validate required parameters
        if not epsilon or not attack_type:
            return jsonify({"error": "Epsilon and attack type are required."}), 400

        # Convert parameters
        epsilon = float(epsilon)
        columns_to_drop = [col.strip() for col in columns_to_drop_input.split(',') if col.strip()]

        # Save uploaded files
        dataset_path = os.path.join(UPLOAD_FOLDER, secure_filename(dataset_file.filename))
        model_path = os.path.join(UPLOAD_FOLDER, secure_filename(model_file.filename))
        scaler_path = os.path.join(UPLOAD_FOLDER, secure_filename(scaler_file.filename))
        encoder_path = os.path.join(UPLOAD_FOLDER, secure_filename(encoder_file.filename))

        dataset_file.save(dataset_path)
        model_file.save(model_path)
        scaler_file.save(scaler_path)
        encoder_file.save(encoder_path)

        # ✅ Ensure TensorFlow session is properly reset
        tf.keras.backend.clear_session()

        # Load files
        df_test = pd.read_csv(dataset_path)
        scaler = joblib.load(scaler_path)
        model = load_model(model_path, compile=False)  # ✅ Prevent potential TF session issues

        # Preprocess data
        X_scaled_test, y_test = preprocess(df_test, scaler_path, columns_to_drop)

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

        X_test_adv = attack_functions[attack_type](classifier, X_scaled_test, epsilon)

        # Denormalize adversarial examples
        X_test_adv_denorm = scaler.inverse_transform(X_test_adv)
        X_test_adv_denorm_df = pd.DataFrame(X_test_adv_denorm, columns=X_scaled_test.columns)

        # Load the label encoder to restore original labels
        encoder = joblib.load(encoder_path)

        # Convert encoded labels back to original class names
        y_test_original = encoder.inverse_transform(y_test.values.ravel())  # Flatten array before decoding

        # Reattach dropped columns
        if columns_to_drop:
            dropped_columns = df_test[columns_to_drop].iloc[X_test_adv_denorm_df.index]
            df_adv_complete = pd.concat([dropped_columns.reset_index(drop=True), X_test_adv_denorm_df.reset_index(drop=True)], axis=1)
        else:
            df_adv_complete = X_test_adv_denorm_df

        # Assign original (not encoded) labels back to the dataset
        df_adv_complete['label'] = y_test_original  # Restore original labels

        # Save adversarial dataset in memory (instead of writing to disk)
        adversarial_dataset_csv = BytesIO()
        df_adv_complete.to_csv(adversarial_dataset_csv, index=False)
        adversarial_dataset_csv.seek(0)  # Move to the beginning of the stream

        # Evaluate metrics
        y_pred_adv = model.predict(X_test_adv)
        y_pred_adv_classes = np.argmax(y_pred_adv, axis=1)
        metrics = evaluation_metrics(y_test.values, y_pred_adv_classes, y_pred_adv, 'Adversarial_Dataset', UPLOAD_FOLDER)

        # Create a JSON response with attack ID and metrics
        json_response = {
            "message": "Attack completed successfully.",
            "attack_id": attack_id,  # Unique attack ID
            "metrics": {k: v for k, v in metrics.items() if k != 'confusion_matrix'},  # Send all except confusion matrix
        }

        # Save metrics JSON in memory
        metrics_json = BytesIO()
        metrics_json.write(json.dumps(json_response, indent=4).encode('utf-8'))
        metrics_json.seek(0)

        # Create a ZIP file containing both the CSV and the JSON response
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"adversarial_dataset_{attack_id}.csv", adversarial_dataset_csv.getvalue())
            zip_file.writestr(f"metrics_{attack_id}.json", metrics_json.getvalue())

        zip_buffer.seek(0)  # Move to the beginning of the ZIP buffer

        # Create a response that contains the ZIP file
        response = make_response(send_file(
            zip_buffer, 
            mimetype='application/zip', 
            as_attachment=True, 
            download_name=f"adversarial_results_{attack_id}.zip"
        ))

        # Attach custom headers
        response.headers["X-Attack-ID"] = attack_id
        response.headers["X-Attack-Status"] = f"Attack ({attack_type}) is successfully operated."

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='192.168.126.144', port=5000, debug=True)  # ✅ Proper server settings
