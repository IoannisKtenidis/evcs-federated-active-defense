<p align="center">
  <img src="white_box/AI4CYBER_logo.png" alt="AAG Logo" width="200"/>
</p>

# 🔐 Adversarial Attack Generator (AAG)

## 📖 Overview
The **Adversarial Attack Generator (AAG)** is a research framework designed to evaluate the robustness of machine learning models against adversarial attacks.  
It integrates **white-box**, **black-box**, and **hybrid (white-box under black-box)** scenarios.

AAG provides:
- **Clean White-Box Attacks** – Standard gradient-based methods (FGSM, PGD, BIM, CW, JSMA).
- **Black-Box Attacks** – Query-based methods (ZOO, HopSkipJump, Boundary, Square, NES).
- **White-Box under Black-Box** – Transferability evaluations across models.
- **Dashboard** - Dashboard for executing adversarial attacks 
- **REST API** – For programmatic execution with ZIP responses (datasets + metrics).

---

## 📂 Repository Structure

``` bash
Adversarial_Attack_Generator/
├─ dataset/ # Datasets for training/testing
│ ├─ CICFlow_meter/
│ │ ├─ Client_1/
│ │ ├─ Client_2/
│ │ ├─ Client_3/
│ │ └─ Full Dataset/
│ │
│ └─ OCPPFlow_meter/
│ ├─ Client_1/
│ ├─ Client_2/
│ ├─ Client_3/
│ ├─ Test.csv
│ └─ Train.csv
│
├─ encoder/ # Encoders (label, categorical)
│ ├─ CICFlow_label_encoder.joblib
│ ├─ encoder_ocpp.joblib
│ └─ python-3.11.0-amd64.exe # (local Python installer, optional)
│
├─ model/ # Pretrained models
│ ├─ CICFlow_model.h5
│ └─ OCPP_model.h5
│
├─ scaler/ # Scalers for preprocessing
│ ├─ CICFlowMeter_scaler.joblib
│ └─ scaler_ocpp.joblib
│
├─ white_box/ # Clean white-box attacks
│ ├─ aag_dashboard.py
│ ├─ white_box_attack.py
│ ├─ utils.py
│ ├─ api_server.py # Flask REST API
│ ├─ swagger_api.py # Swagger/OpenAPI definitions
│ └─ README.md # Detailed White-Box documentation
│
├─ black_box/ # Black-box & hybrid attacks
│ ├─ black_box_attack_other.py
│ ├─ black_box_attacks.py
│ ├─ attack_type.py
│ └─ README.md # Detailed Black-Box documentation
│
├─ requirements.txt # Dependencies
└─ README.md # General overview
---
```

---

## 📊 Supported Attacks

### ✅ White-Box Attacks
- FGSM *(Fast Gradient Sign Method)*
- PGD *(Projected Gradient Descent)*
- BIM *(Basic Iterative Method)*
- JSMA *(Jacobian Saliency Map Attack)*
- CW *(Carlini & Wagner)*

### ✅ Black-Box Attacks
- ZOO *(Zeroth-Order Optimization)*
- HopSkipJump (HSJ)
- Boundary Attack

### ✅ Hybrid
- **White-Box under Black-Box**: transferring adversarial samples between surrogate and target models.

---

## 📘 Documentation

- 📄 [White-Box Attacks Documentation](white_box/README.md)  
- 📄 [Black-Box Attacks Documentation](black_box/README.md)  

---

## 🚀 Quickstart

### Install dependencies
```bash
pip install -r requirements.txt

