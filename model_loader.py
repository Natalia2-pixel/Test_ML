import os
import gdown
import zipfile
import joblib
import tensorflow as tf
from functools import lru_cache

@lru_cache(maxsize=1)
def download_and_extract_models_from_drive(file_id):
    output_zip = "models.zip"
    extract_to = "Models"

    if not os.path.exists(extract_to):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_zip, quiet=False)

        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        os.remove(output_zip)

    return extract_to

@lru_cache(maxsize=1)
def load_models():
    zip_file_id = "1DN8GSYUkVaE5b6SaK4GJYjrhCja612Av"
    model_dir = download_and_extract_models_from_drive(zip_file_id)

    svm_model = joblib.load(f"{model_dir}/svm_model.pkl")
    gbm_model = joblib.load(f"{model_dir}/gbm_model.pkl")
    pca_x = joblib.load(f"{model_dir}/pca_x.pkl")
    pca_y = joblib.load(f"{model_dir}/pca_y.pkl")
    cnn_model = tf.keras.models.load_model(f"{model_dir}/cnn_model.h5", compile=False)
    cnn_model.compile(optimizer='adam', loss='mse')
    xception_model = tf.keras.models.load_model(f"{model_dir}/xception_model.h5", compile=False)
    xception_model.compile(optimizer='adam', loss='mse')
    cgan_generator = tf.keras.models.load_model(f"{model_dir}/cgan_generator_model.h5", compile=False)
    cgan_generator.compile(optimizer='adam', loss='mse')

    return svm_model, gbm_model, pca_x, pca_y, cnn_model, xception_model, cgan_generator
