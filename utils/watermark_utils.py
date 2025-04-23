import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from model_loader import load_models

# Load all models using the cached loader (no Streamlit UI in this file)
svm_model, gbm_model, pca_x, pca_y, scaler_x, scaler_y, cnn_model, cgan_generator = load_models()

def apply_watermark_ml_model(cover_image, watermark_image=None, model_type="SVM", alpha=0.3):
    if cover_image is None:
        raise ValueError("Cover image is missing.")
    if watermark_image is None:
        watermark_image = Image.new("L", (64, 64), color=0)

    cover = cover_image.convert("L").resize((64, 64))
    mark = watermark_image.convert("L").resize((64, 64))

    cover_np = np.array(cover, dtype=np.float32) / 255.0
    mark_np = np.array(mark, dtype=np.float32) / 255.0

    blended_gt = (1 - alpha) * cover_np + alpha * mark_np
    input_vec = np.hstack([cover_np.flatten(), mark_np.flatten()])

    if input_vec.shape[0] != pca_x.components_.shape[1]:
        raise ValueError(f"Input vector length mismatch for PCA: {input_vec.shape[0]}")

    # Apply scaler and PCA as done in training
    scaled_input = scaler_x.transform([input_vec])
    reduced_input = pca_x.transform(scaled_input)

    if model_type == "SVM":
        predicted = svm_model.predict(reduced_input)
    elif model_type == "GBM":
        predicted = gbm_model.predict(reduced_input)
    else:
        raise ValueError("Unknown ML model type")

    # Inverse PCA and scaler for output
    output_scaled = pca_y.inverse_transform(predicted)
    reconstructed = scaler_y.inverse_transform(output_scaled)[0]

    predicted_image = reconstructed.reshape(64, 64)

    mse = np.mean((blended_gt - predicted_image) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    ssim_score = ssim(blended_gt, predicted_image, data_range=1.0)

    # Normalize output image for better visualization
    norm_img = (predicted_image - predicted_image.min()) / (predicted_image.max() - predicted_image.min() + 1e-8)
    output_img = Image.fromarray((norm_img * 255).astype(np.uint8))

    return output_img, {
    "mse": mse,
    "psnr": psnr,
    "ssim": ssim_score
}

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

    elif model_type == "GAN":
        stacked = np.stack([cover_np, mark_np], axis=-1)
        input_tensor = tf.convert_to_tensor(stacked[np.newaxis, ...])
        predicted = cgan_generator.predict(input_tensor, verbose=0)[0].squeeze()

    else:
        raise ValueError("Unsupported DL model selected")

    mse = np.mean((blended_gt - predicted) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    ssim_score = ssim(blended_gt, predicted, data_range=1.0)

   # Normalize for better visibility (optional but highly recommended)
    norm_img = (predicted - predicted.min()) / (predicted.max() - predicted.min() + 1e-8)
    output_img = Image.fromarray((norm_img * 255).astype(np.uint8))

    return output_img, {
        "mse": mse, "psnr": psnr, "ssim": ssim_score
    }

def apply_watermark_with_model(model, cover_image, custom_watermark_img=None):
    if cover_image is None:
        raise ValueError("Cover image is missing. Please upload one.")
    if model in ["SVM", "GBM"]:
        return apply_watermark_ml_model(cover_image, custom_watermark_img, model_type=model)
    elif model in ["CNN", "GAN"]:
        return apply_watermark_dl_model(cover_image, custom_watermark_img, model_type=model)
    else:
        raise ValueError("Unknown model selected. Please choose from: SVM, GBM, CNN, GAN")
