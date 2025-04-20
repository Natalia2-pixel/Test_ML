import os
import requests
import joblib
import tensorflow as tf
from functools import lru_cache

# Dictionary of model file names and their raw GitHub URLs
model_files = {
    "svm_model": "https://github.com/Natalia2-pixel/Test_ML/raw/main/svm_model.pkl",
    "gbm_model": "https://github.com/Natalia2-pixel/Test_ML/raw/main/gbm_model.pkl",
    "pca_x": "https://github.com/Natalia2-pixel/Test_ML/raw/main/pca_x.pkl",
    "pca_y": "https://github.com/Natalia2-pixel/Test_ML/raw/main/pca_y.pkl",
    "cnn_model": "https://github.com/Natalia2-pixel/Test_ML/raw/main/cnn_model.h5",
    "cgan_generator_model": "https://github.com/Natalia2-pixel/Test_ML/raw/main/cgan_generator_model.h5",
}

def download_model_file(name, url, dest_folder="."):
    os.makedirs(dest_folder, exist_ok=True)
    file_path = os.path.join(dest_folder, os.path.basename(url))

    if not os.path.exists(file_path):
        print(f"Downloading {name}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"{name} downloaded ✅")
        else:
            raise Exception(f"Failed to download {name}: HTTP {response.status_code}")

    return file_path

@lru_cache(maxsize=1)
def load_models():
    paths = {name: download_model_file(name, url) for name, url in model_files.items()}

    # Load ML models
    svm_model = joblib.load(paths["svm_model"])
    gbm_model = joblib.load(paths["gbm_model"])
    pca_x = joblib.load(paths["pca_x"])
    pca_y = joblib.load(paths["pca_y"])

    # Load DL models
    cnn_model = tf.keras.models.load_model(paths["cnn_model"], compile=False)
    cnn_model.compile(optimizer='adam', loss='mse')

    cgan_generator = tf.keras.models.load_model(paths["cgan_generator_model"], compile=False)
    cgan_generator.compile(optimizer='adam', loss='mse')

    return svm_model, gbm_model, pca_x, pca_y, cnn_model, cgan_generator
