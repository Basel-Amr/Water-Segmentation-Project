import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
import io, base64, os

def preprocess_9ch_for_inference(image_path):
    with rasterio.open(image_path) as src:
        img = src.read().astype(np.float32)

    blue, green, red = img[1], img[2], img[3]
    band8, band11, band12 = img[7], img[10], img[11]
    ndwi = (green - band8) / (green + band8 + 1e-7)
    mndwi = (green - band11) / (green + band11 + 1e-7)
    awei = 4 * (green - band11) - (0.25 * band8 + 2.75 * band11)
    image = np.stack([blue, green, red, band8, band11, band12, ndwi, mndwi, awei], axis=0)
    image = (image - image.mean(axis=(1, 2), keepdims=True)) / \
            (image.std(axis=(1, 2), keepdims=True) + 1e-7)
    image_resized = np.stack([cv2.resize(ch, (128, 128)) for ch in image], axis=0)
    return image_resized[np.newaxis, :, :, :]

def visualize_single_prediction(model_session, image_path):
    img_input = preprocess_9ch_for_inference(image_path)
    outputs = model_session.run(None, {model_session.get_inputs()[0].name: img_input})
    pred_mask = outputs[0][0, 0]
    pred_mask_bin = pred_mask > 0.5

    rgb = img_input[0, [0, 1, 2], :, :]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    # Overlay
    overlay = rgb.copy()
    overlay[pred_mask_bin] = [1, 0, 0]  # red overlay
    combined_overlay = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    # Ground Truth
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    gt_path = os.path.join("data", "labels", f"{base_name}.png")
    ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
    if ground_truth is not None:
        ground_truth = cv2.resize(ground_truth, (128, 128))

    # Visualization
    fig, axs = plt.subplots(1, 4 if ground_truth is not None else 3, figsize=(24, 8))
    axs[0].imshow(rgb)
    axs[0].set_title("🌍 RGB", fontsize=22, color="#00e0ff", fontweight='bold')
    axs[0].axis("off")

    axs[1].imshow(combined_overlay)
    axs[1].set_title("Overlay", fontsize=22, color="#00ff88", fontweight='bold')
    axs[1].axis("off")

    if ground_truth is not None:
        axs[2].imshow(ground_truth, cmap="gray")
        axs[2].set_title("Ground Truth", fontsize=22, color="#ffdd00", fontweight='bold')
        axs[2].axis("off")
        axs[3].imshow(pred_mask_bin, cmap="gray")
        axs[3].set_title("Predicted Mask", fontsize=22, color="#00ff88", fontweight='bold')
        axs[3].axis("off")
    else:
        axs[2].imshow(pred_mask_bin, cmap="gray")
        axs[2].set_title("Predicted Mask", fontsize=22, color="#00ff88", fontweight='bold')
        axs[2].axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}", pred_mask_bin
