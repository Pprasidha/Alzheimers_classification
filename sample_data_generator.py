"""
Realistic Brain MRI Simulation Generator for Alzheimer's Disease Stages
Generates anatomically consistent synthetic axial MRI brain slices:
  - NonDemented: Full cortical volume, small lateral ventricles, dense temporal lobes.
  - VeryMildDemented: Mild enlargement of lateral ventricles, slight cortical margin changes.
  - MildDemented: Clear dilation of lateral ventricles, noticeable hippocampal atrophy markers.
  - ModerateDemented: Pronounced ventriculomegaly (dilated ventricles), widespread cerebral atrophy.

Used for instant testing, unit testing, and immediate end-to-end pipeline validation.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def generate_brain_mri_slice(stage_idx, size=(224, 224), seed=None):
    """
    Synthesize an axial brain MRI slice simulating specific structural Alzheimer's biomarkers.
    stage_idx:
      0: MildDemented
      1: ModerateDemented
      2: NonDemented
      3: VeryMildDemented
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = size
    # 0 = background (black)
    img_arr = np.zeros((h, w), dtype=np.float32)

    # Grid coordinates
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2

    # 1. Cranium / Skull Oval
    skull_rx, skull_ry = int(w * 0.42), int(h * 0.45)
    skull_mask = ((x - cx)**2 / (skull_rx**2) + (y - cy)**2 / (skull_ry**2)) <= 1.0
    
    # Bone intensity
    img_arr[skull_mask] = 0.85 + np.random.uniform(-0.05, 0.05, skull_mask.shape)[skull_mask]

    # 2. Meninges / Subarachnoid space (CSF ring, dark)
    meninges_rx, meninges_ry = int(w * 0.40), int(h * 0.43)
    meninges_mask = ((x - cx)**2 / (meninges_rx**2) + (y - cy)**2 / (meninges_ry**2)) <= 1.0
    img_arr[meninges_mask] = 0.12

    # 3. Brain Parenchyma (Cortex + White/Gray Matter)
    # Brain volume shrinks as dementia progresses
    atrophy_factors = {
        2: 1.0,    # NonDemented: Full volume
        3: 0.96,   # VeryMild: Minimal shrinkage
        0: 0.91,   # Mild: Noticeable shrinkage
        1: 0.84    # Moderate: Severe shrinkage
    }
    atrophy = atrophy_factors[stage_idx]

    brain_rx, brain_ry = int(w * 0.38 * atrophy), int(h * 0.41 * atrophy)
    brain_mask = ((x - cx)**2 / (brain_rx**2) + (y - cy)**2 / (brain_ry**2)) <= 1.0

    # Base brain tissue intensity (Gray matter ~0.55, White matter ~0.70)
    tissue = np.full((h, w), 0.58, dtype=np.float32)
    # Add natural anatomical texture
    noise = np.random.normal(0, 0.03, (h, w))
    tissue += noise
    img_arr[brain_mask] = tissue[brain_mask]

    # White matter core (central oval)
    wm_rx, wm_ry = int(brain_rx * 0.65), int(brain_ry * 0.65)
    wm_mask = ((x - cx)**2 / (wm_rx**2) + (y - cy)**2 / (wm_ry**2)) <= 1.0
    img_arr[wm_mask] = 0.72 + np.random.uniform(-0.03, 0.03, wm_mask.shape)[wm_mask]

    # 4. Lateral Ventricles (Key Biomarker in Alzheimer's Disease)
    # In Alzheimer's, lateral ventricles dilate significantly (Ventriculomegaly)
    ventricle_scales = {
        2: (0.04, 0.10),   # NonDemented: Slender, small
        3: (0.07, 0.14),   # VeryMild: Slightly widening
        0: (0.11, 0.20),   # Mild: Dilated
        1: (0.18, 0.28)    # Moderate: Severely dilated butterfly shape
    }
    vrx_factor, vry_factor = ventricle_scales[stage_idx]
    vrx = int(w * vrx_factor)
    vry = int(h * vry_factor)

    # Left and Right ventricles
    offset_x = int(w * 0.06 + vrx * 0.35)
    left_ventricle = ((x - (cx - offset_x))**2 / (vrx**2) + (y - cy)**2 / (vry**2)) <= 1.0
    right_ventricle = ((x - (cx + offset_x))**2 / (vrx**2) + (y - cy)**2 / (vry**2)) <= 1.0
    
    # Third ventricle (central slit)
    third_ventricle = (np.abs(x - cx) <= int(w * 0.015 * (1 + stage_idx * 0.5))) & (np.abs(y - cy) <= int(h * 0.12))

    # Ventricles contain CSF (Cerebrospinal Fluid) which is hypointense (dark) on T1-weighted MRI
    img_arr[left_ventricle | right_ventricle | third_ventricle] = 0.08 + np.random.uniform(0, 0.04)

    # 5. Sulci / Cortical fissures (deepening fissures in moderate/severe atrophy)
    sulcal_num = [4, 8, 14, 22][stage_idx if stage_idx in [0, 1, 2, 3] else 0]
    angles = np.linspace(0, 2 * np.pi, sulcal_num, endpoint=False)
    for ang in angles:
        r_start = brain_rx * 0.75
        r_end = brain_rx * 1.05
        sx = cx + int(r_start * np.cos(ang))
        sy = cy + int(r_start * np.sin(ang))
        ex = cx + int(r_end * np.cos(ang))
        ey = cy + int(r_end * np.sin(ang))
        # Draw small dark fissure
        rr = np.hypot(x - cx, y - cy)
        theta = np.arctan2(y - cy, x - cx)
        fissure_mask = (np.abs(theta - ang) < 0.02) & (rr >= r_start) & (rr <= r_end)
        img_arr[fissure_mask] = 0.10

    # Clip to [0, 1]
    img_arr = np.clip(img_arr, 0.0, 1.0)
    uint8_img = np.uint8(img_arr * 255)

    pil_img = Image.fromarray(uint8_img)
    # Subtle Gaussian smoothing to simulate clinical scanner point-spread function (PSF)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))

    return pil_img


def generate_synthetic_dataset(output_dir="data/mri_dataset", samples_per_class=40):
    """
    Generate balanced dataset folders for quick training and testing.
    """
    class_map = {
        "MildDemented": 0,
        "ModerateDemented": 1,
        "NonDemented": 2,
        "VeryMildDemented": 3
    }

    os.makedirs(output_dir, exist_ok=True)
    total_generated = 0

    for folder_name, stage_idx in class_map.items():
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for i in range(samples_per_class):
            img = generate_brain_mri_slice(stage_idx, size=(224, 224), seed=stage_idx * 1000 + i)
            file_name = f"scan_{stage_idx}_{i:04d}.jpg"
            img.save(os.path.join(folder_path, file_name), quality=95)
            total_generated += 1

    print(f"[OK] Generated {total_generated} synthetic MRI scans across 4 classes in '{output_dir}'.")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic MRI dataset for Alzheimer's stage classification.")
    parser.add_argument("--out", type=str, default="data/mri_dataset", help="Output directory")
    parser.add_argument("--samples", type=int, default=50, help="Number of scans per stage")
    args = parser.parse_args()

    generate_synthetic_dataset(args.out, samples_per_class=args.samples)
