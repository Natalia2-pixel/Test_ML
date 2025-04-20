import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import joblib
import tensorflow as tf
from model_loader import load_models

# Load all models using the cached loader (no Streamlit UI in this file)
svm_model, gbm_model, pca_x, pca_y, cnn_model, xception_model, cgan_generator = load_models()

def apply_watermark_ml_model(cover_image, watermark_image=None, model_type="SVM", alpha=0.3):
    if cover_image is None:
        raise ValueError("Cover image is missing.")
    if watermark_image is None:
        watermark_image = Image.new("L", (64, 64), color=0)

    # Convert images to grayscale and resize
    cover = cover_image.convert("L").resize((64, 64))
    mark = watermark_image.convert("L").resize((64, 64))

    # Normalize arrays
    cover_np = np.array(cover, dtype=np.float32) / 255.0
    mark_np = np.array(mark, dtype=np.float32) / 255.0

    # Blend ground truth
    blended_gt = (1 - alpha) * cover_np + alpha * mark_np
    input_vec = np.hstack([cover_np.flatten(), mark_np.flatten()])

    # PCA input dimension check
    if input_vec.shape[0] != pca_x.components_.shape[1]:
        raise ValueError(f"Input vector length mismatch for PCA: {input_vec.shape[0]}")

    reduced_input = pca_x.transform([input_vec])

    if model_type == "SVM":
        predicted = svm_model.predict(reduced_input)
    elif model_type == "GBM":
        predicted = gbm_model.predict(reduced_input)
    else:
        raise ValueError("Unknown ML model type")

    reconstructed = pca_y.inverse_transform(predicted)[0]
    predicted_image = reconstructed.reshape(64, 64)

    # Evaluation
    mse = np.mean((blended_gt - predicted_image) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    ssim_score = ssim(blended_gt, predicted_image, data_range=1.0)

    return Image.fromarray((predicted_image * 255).astype(np.uint8)), {"mse": mse, "psnr": psnr, "ssim": ssim_score}

def apply_watermark_dl_model(cover_image, watermark_image=None, model_type="CNN", alpha=0.3):
    if cover_image is None:
        raise ValueError("Cover image is missing.")
    if watermark_image is None:
        watermark_image = Image.new("L", (64, 64), color=0)

    cover = cover_image.convert("L").resize((64, 64))
    mark = watermark_image.convert("L").resize((64, 64))

    cover_np = np.array(cover, dtype=np.float32) / 255.0
    mark_np = np.array(mark, dtype=np.float32) / 255.0
    blended_gt = (1 - alpha) * cover_np + alpha * mark_np

    if model_type == "CNN":
        stacked = np.stack([cover_np, mark_np], axis=-1)
        input_tensor = np.expand_dims(stacked, axis=0)
        predicted = cnn_model.predict(input_tensor, verbose=0)[0].squeeze()

    elif model_type == "Xception":
        rgb_tensor = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cover_np[..., np.newaxis]))
        resized = tf.image.resize(rgb_tensor, [71, 71])
        input_tensor = tf.expand_dims(resized, axis=0)
        predicted = xception_model.predict(input_tensor, verbose=0)[0]
        predicted = tf.image.resize(predicted, [64, 64]).numpy().squeeze()

    elif model_type == "GAN":
        stacked = np.stack([cover_np, mark_np], axis=-1)
        input_tensor = tf.convert_to_tensor(stacked[np.newaxis, ...])
        predicted = cgan_generator.predict(input_tensor, verbose=0)[0].squeeze()

    else:
        raise ValueError("Unsupported DL model selected")

    mse = np.mean((blended_gt - predicted) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    ssim_score = ssim(blended_gt, predicted, data_range=1.0)

    return Image.fromarray((predicted * 255).astype(np.uint8)), {"mse": mse, "psnr": psnr, "ssim": ssim_score}

def apply_watermark_with_model(model, cover_image, custom_watermark_img=None):
    if cover_image is None:
        raise ValueError("Cover image is missing. Please upload one.")
    if model in ["SVM", "GBM"]:
        return apply_watermark_ml_model(cover_image, custom_watermark_img, model_type=model)
    elif model in ["CNN", "Xception", "GAN"]:
        return apply_watermark_dl_model(cover_image, custom_watermark_img, model_type=model)
    else:
        raise ValueError("Unknown model selected")