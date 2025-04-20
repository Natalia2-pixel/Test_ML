from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage.transform import resize

def evaluate_prediction(true_img, pred_img):
    true_flat = true_img.flatten()
    pred_flat = pred_img.flatten()

    # If predicted and actual sizes mismatch, align them safely
    if len(true_flat) != len(pred_flat):
        min_len = min(len(true_flat), len(pred_flat))
        true_flat = true_flat[:min_len]
        pred_flat = pred_flat[:min_len]

    mse_val = mean_squared_error(true_flat, pred_flat)

    # Try to calculate SSIM and PSNR (if shapes compatible)
    try:
        if pred_img.shape != true_img.shape:
            # Resize prediction to match original shape for SSIM
            pred_img_resized = resize(pred_img, true_img.shape, preserve_range=True)
        else:
            pred_img_resized = pred_img

        psnr_val = 10 * np.log10(1.0 / mse_val)
        ssim_val = ssim(true_img, pred_img_resized, data_range=1.0)
    except Exception:
        psnr_val = 0.0
        ssim_val = 0.0

    return mse_val, psnr_val, ssim_val
