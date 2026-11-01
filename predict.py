"""
Standalone Inference and Explainability (Grad-CAM) CLI
Classifies a single brain MRI scan and produces an attention heatmap overlay.
"""

import os
import argparse
import torch
from PIL import Image
import numpy as np

from models.cnn_model import AlzheimerCNN
from models.explainability import GradCAM, overlay_gradcam
from utils.dataset import get_transforms, DISPLAY_NAMES, CLASS_NAMES


def predict_scan(image_path,
                 checkpoint_path="models/checkpoints/best_alzheimer_model.pth",
                 output_cam_path=None):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AlzheimerCNN(in_channels=3, num_classes=4).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded weights from {checkpoint_path}")
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}. Running with initialized weights.")
    model.eval()

    # Preprocess image
    orig_img = Image.open(image_path).convert('RGB')
    transform = get_transforms(img_size=224, is_train=False)
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Initialize Grad-CAM on the last attention convolutional stage
    grad_cam = GradCAM(model, target_layer=model.grad_cam_layer.conv2)

    # Forward + Grad-CAM
    heatmap, pred_class, logits = grad_cam.generate_heatmap(input_tensor)
    grad_cam.remove_hooks()

    probabilities = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    confidence = probabilities[pred_class] * 100.0

    print("\n" + "="*50)
    print("           MRI SCAN DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Scan File:        {image_path}")
    print(f"Predicted Stage:  {DISPLAY_NAMES[pred_class]}")
    print(f"Confidence:       {confidence:.2f}%\n")
    print("Class Probability Distribution:")
    for idx, (cls_name, prob) in enumerate(zip(DISPLAY_NAMES, probabilities)):
        bar = "#" * int(prob * 30)
        marker = "<-- [PREDICTED]" if idx == pred_class else ""
        print(f"  {cls_name:<38} : {prob*100:5.2f}% |{bar:<30}| {marker}")

    # Generate overlay
    cam_overlay = overlay_gradcam(orig_img, heatmap, alpha=0.45)
    if output_cam_path is None:
        base, ext = os.path.splitext(image_path)
        output_cam_path = f"{base}_gradcam{ext}"

    cam_overlay.save(output_cam_path)
    print(f"\n[OK] Grad-CAM attention map saved to: {output_cam_path}")

    return {
        "pred_class": pred_class,
        "class_name": DISPLAY_NAMES[pred_class],
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
        "gradcam_path": output_cam_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Alzheimer's stage from MRI scan.")
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image scan")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_alzheimer_model.pth")
    parser.add_argument("--out_cam", type=str, default=None, help="Output path for Grad-CAM overlay")
    args = parser.parse_args()

    predict_scan(args.image, args.checkpoint, args.out_cam)
