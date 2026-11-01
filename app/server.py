"""
Flask Web Application for Alzheimer's MRI Disease Stage Classification
Provides real-time scan diagnosis, probability graphs, and Grad-CAM interpretability.
"""

import os
import io
import base64
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np

import sys
# Add parent dir to path so models and utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cnn_model import AlzheimerCNN
from models.explainability import GradCAM, overlay_gradcam
from utils.dataset import get_transforms, DISPLAY_NAMES, CLASS_NAMES

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'best_alzheimer_model.pth'))

# Model loader
_model_instance = None

def get_or_load_model():
    global _model_instance
    if _model_instance is None:
        model = AlzheimerCNN(in_channels=3, num_classes=4).to(DEVICE)
        if os.path.exists(CHECKPOINT_PATH):
            try:
                ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"[SERVER] Loaded model checkpoint from {CHECKPOINT_PATH}")
            except Exception as e:
                print(f"[SERVER WARNING] Could not load checkpoint: {e}")
        else:
            print("[SERVER] Checkpoint not found, using initialized weights for demonstration.")
        model.eval()
        _model_instance = model
    return _model_instance


CLINICAL_INSIGHTS = {
    0: {
        "title": "Mild Impairment (Mild Demented)",
        "badge": "warning",
        "description": "Noticeable memory loss and spatial disorientation. MRI typically reveals widening lateral ventricles and cortical thinning, particularly around the temporal and parietal lobes.",
        "recommendation": "Consult a neurologist for neuropsychological evaluation, cognitive assessment (MMSE/MoCA), and biomarker tracking."
    },
    1: {
        "title": "Moderate Impairment (Moderate Demented)",
        "badge": "danger",
        "description": "Significant cognitive and functional impairment. MRI scans exhibit severe ventriculomegaly (dilated ventricles) and extensive diffuse cortical cerebral atrophy.",
        "recommendation": "Comprehensive clinical care plan, high supervision, targeted medical interventions, and family caregiver support."
    },
    2: {
        "title": "Normal Control (Non-Demented)",
        "badge": "success",
        "description": "Structural brain volume is within age-adjusted normal limits. Slender lateral ventricles and preserved hippocampal volume without pathological atrophy.",
        "recommendation": "Maintain regular health checkups, cardiovascular fitness, and cognitive stimulation routines."
    },
    3: {
        "title": "Early Impairment (Very Mild Demented)",
        "badge": "info",
        "description": "Subtle, borderline cognitive changes often corresponding to early Mild Cognitive Impairment (MCI). Microstructural changes localized to entorhinal cortex.",
        "recommendation": "Schedule follow-up MRI in 6-12 months for longitudinal atrophy tracking and early lifestyle/therapeutic interventions."
    }
}


def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files and 'sample_path' not in request.form:
        return jsonify({'error': 'No file or sample provided'}), 400

    try:
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        elif 'sample_path' in request.form:
            sample_rel = request.form['sample_path']
            sample_full = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', sample_rel))
            if not os.path.exists(sample_full):
                return jsonify({'error': 'Sample file not found'}), 404
            image = Image.open(sample_full).convert('RGB')
        else:
            return jsonify({'error': 'Invalid request'}), 400

        model = get_or_load_model()
        transform = get_transforms(img_size=224, is_train=False)
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Grad-CAM explanation
        grad_cam = GradCAM(model, target_layer=model.grad_cam_layer.conv2)
        heatmap, pred_class, logits = grad_cam.generate_heatmap(input_tensor)
        grad_cam.remove_hooks()

        probabilities = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        confidence = float(probabilities[pred_class] * 100.0)

        # Create Grad-CAM overlay
        cam_overlay = overlay_gradcam(image, heatmap, alpha=0.45)

        orig_b64 = image_to_base64(image.resize((300, 300)))
        overlay_b64 = image_to_base64(cam_overlay.resize((300, 300)))

        prob_list = [
            {
                "stage": DISPLAY_NAMES[i],
                "short_name": CLASS_NAMES[i],
                "probability": round(float(probabilities[i] * 100.0), 2)
            }
            for i in range(4)
        ]

        insights = CLINICAL_INSIGHTS.get(pred_class, {})

        return jsonify({
            "status": "success",
            "predicted_class": pred_class,
            "predicted_stage": DISPLAY_NAMES[pred_class],
            "confidence": round(confidence, 2),
            "probabilities": prob_list,
            "clinical_insights": insights,
            "original_image_base64": f"data:image/jpeg;base64,{orig_b64}",
            "gradcam_image_base64": f"data:image/jpeg;base64,{overlay_b64}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Alzheimer's Disease Diagnostic Web App on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
