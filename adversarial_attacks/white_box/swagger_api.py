from flask import Flask, request, jsonify, send_file
from flasgger import Swagger
import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from io import BytesIO
import zipfile
import json
import uuid
import matplotlib

from white_box_attack import (
    preprocess, evaluation_metrics,
    fgsm_attack, pgd_attack, bim_attack,
    jsma_attack, cw_attack
)
from art.estimators.classification import KerasClassifier

# Non-GUI backend for matplotlib
matplotlib.use('Agg')

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# App initialization
app = Flask(__name__)
swagger = Swagger(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/')
def index():
    """
    Welcome message for the API.
    ---
    responses:
      200:
        description: API is running.
    """
    return jsonify({
        "message": "Welcome to the Adversarial Attack API.",
        "endpoints": {
            "POST /run_attack": "Run an adversarial attack and get a ZIP file.",
        }
    })


@app.route('/run_attack', methods=['POST'])
def run_attack():
    """
    Run an adversarial attack and get back a ZIP with dataset + metrics.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: dataset
        in: formData
        type: file
        required: true
      - name: model
        in: formData
        type: file
        required: true
      - name: scaler
        in: formData
        type: file
        required: true
      - name: encoder
        in: formData
        type: file
        required: true
      - name: epsilon
        in: formData
        type: number
        required: true
      - name: attack
        in: formData
        type: string
        enum: [fgsm, pgd, bim, jsma, cw]
        required: true
      - name: columns_to_drop
        in: formData
        type: string
        required: false
        description: Comma-separated column names to drop
    responses:
      200:
        description: ZIP file containing adversarial dataset and metrics
        content:
          application/zip:
            schema:
              type: string
              format: binary
      400:
        description: Missing input or invalid parameter
      500:
        description: Internal server error
    """
    try:
        attack_id = str(uuid.uuid4())[:8]

        dataset_file = request.files.get('dataset')
        model_file = request.files.get('model')
        scaler_file = request.files.get('scaler')
        encoder_file = request.files.get('encoder')
        epsilon = request.form.get('epsilon')
        attack_type = request.form.get('attack')
        columns_to_drop_input = request.form.get('columns_to_drop', '')

        if not all([dataset_file, model_file, scaler_file, encoder_file, epsilon, attack_type]):
            return jsonify({"error": "Missing one or more required inputs"}), 400

        epsilon = float(epsilon)
        columns_to_drop = [c.strip() for c in columns_to_drop_input.split(',') if c.strip()]

        # Save uploaded files
        dataset_path = os.path.join(UPLOAD_FOLDER, secure_filename(dataset_file.filename))
        model_path = os.path.join(UPLOAD_FOLDER, secure_filename(model_file.filename))
        scaler_path = os.path.join(UPLOAD_FOLDER, secure_filename(scaler_file.filename))
        encoder_path = os.path.join(UPLOAD_FOLDER, secure_filename(encoder_file.filename))

        dataset_file.save(dataset_path)
        model_file.save(model_path)
        scaler_file.save(scaler_path)
        encoder_file.save(encoder_path)

        # Load inputs
        tf.keras.backend.clear_session()
        df_test = pd.read_csv(dataset_path)
        model = load_model(model_path, compile=False)
        X_scaled_test, y_test = preprocess(df_test, scaler_path, columns_to_drop)
        classifier = KerasClassifier(model=model, clip_values=(0, 1))

        # Run attack
        attack_map = {
            'fgsm': fgsm_attack,
            'pgd': pgd_attack,
            'bim': bim_attack,
            'jsma': lambda clf, x, eps: jsma_attack(clf, x, theta=1.0, gamma=0.1),
            'cw': cw_attack
        }

        if attack_type not in attack_map:
            return jsonify({"error": f"Unsupported attack type: {attack_type}"}), 400

        X_test_adv = attack_map[attack_type](classifier, X_scaled_test, epsilon)

        # Post-attack processing
        scaler = joblib.load(scaler_path)
        X_test_adv_denorm = scaler.inverse_transform(X_test_adv)
        df_adv = pd.DataFrame(X_test_adv_denorm, columns=X_scaled_test.columns)

        encoder = joblib.load(encoder_path)
        df_adv['label'] = encoder.inverse_transform(y_test.values.ravel())

        # Evaluate
        y_pred_adv = model.predict(X_test_adv)
        y_pred_classes = np.argmax(y_pred_adv, axis=1)
        metrics = evaluation_metrics(y_test.values, y_pred_classes, y_pred_adv, f'attack_eval_{attack_id}', OUTPUT_FOLDER)

        # ZIP everything
        zip_stream = BytesIO()
        with zipfile.ZipFile(zip_stream, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"adversarial_dataset_{attack_id}.csv", df_adv.to_csv(index=False))
            zf.writestr(f"metrics_{attack_id}.json", json.dumps(metrics, indent=4))

        zip_stream.seek(0)
        return send_file(
            zip_stream,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"adversarial_results_{attack_id}.zip"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='192.168.126.144', port=5001, debug=True)
