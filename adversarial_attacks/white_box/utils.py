import json 
import os 
from keras.models import load_model
import joblib

def load_config(config_path):

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as config_file: 
        config = json.load(config_file)
    return config
