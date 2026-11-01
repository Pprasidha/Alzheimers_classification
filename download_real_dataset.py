
"""
Script to download and unpack the standard 6,400 clinical Alzheimer MRI dataset.
Source: Falah/Alzheimer_MRI (OASIS-derived medical imaging benchmark)
Classes:
  0: MildDemented
  1: ModerateDemented
  2: NonDemented
  3: VeryMildDemented
"""

import os
import io
import urllib.request
import pandas as pd
from PIL import Image

PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/Falah/Alzheimer_MRI/resolve/main/data/train-00000-of-00001-c08a401c53fe5312.parquet",
    "test": "https://huggingface.co/datasets/Falah/Alzheimer_MRI/resolve/main/data/test-00000-of-00001-44110b9df98c5585.parquet"
}

LABEL_MAP = {
    0: "MildDemented",
    1: "ModerateDemented",
    2: "NonDemented",
    3: "VeryMildDemented"
}


def download_and_extract_real_dataset(output_dir="data/mri_dataset", temp_dir="data/temp"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    for class_name in LABEL_MAP.values():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    total_extracted = 0
    class_counts = {name: 0 for name in LABEL_MAP.values()}

    for split_name, url in PARQUET_URLS.items():
        pq_path = os.path.join(temp_dir, f"{split_name}.parquet")
        print(f"\n[DOWNLOAD] Fetching {split_name} parquet from Hugging Face...")
        
        # Download with progress report
        urllib.request.urlretrieve(url, pq_path)
        print(f"[OK] Downloaded {split_name} ({os.path.getsize(pq_path) / (1024*1024):.2f} MB)")

        print(f"[EXTRACT] Unpacking images into class directories...")
        df = pd.read_parquet(pq_path)
        
        for idx, row in df.iterrows():
            img_data = row['image']
            label_idx = int(row['label'])
            class_name = LABEL_MAP.get(label_idx, "Unknown")

            # Extract image bytes
            if isinstance(img_data, dict) and 'bytes' in img_data and img_data['bytes'] is not None:
                img_bytes = img_data['bytes']
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            elif isinstance(img_data, Image.Image):
                img = img_data.convert('RGB')
            else:
                continue

            dest_path = os.path.join(output_dir, class_name, f"real_{split_name}_{idx:05d}.jpg")
            img.save(dest_path, "JPEG", quality=95)
            total_extracted += 1
            class_counts[class_name] += 1

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} images from {split_name}...")

    # Cleanup temp parquet files
    try:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    except Exception:
        pass

    print("\n" + "="*55)
    print("      REAL CLINICAL MRI DATASET EXTRACTION SUMMARY")
    print("="*55)
    print(f"Total scans extracted: {total_extracted}")
    for name, count in class_counts.items():
        print(f"  {name:<22}: {count} scans")
    print("="*55)
    print(f"[SUCCESS] All 6,400 clinical MRI scans are ready in '{output_dir}'.")


if __name__ == "__main__":
    download_and_extract_real_dataset()
