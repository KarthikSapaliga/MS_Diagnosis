import os
from pathlib import Path
import argparse
import nibabel as nib
import numpy as np

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True, help="Path to dataset folder")
    parser.add_argument('--ref_template', type=Path, default="./resources/MNI152_T1_1mm.nii.gz", help="Path to the MNI152 template")
    parser.add_argument('--output_dir', type=Path, default="./Processed", help="Output directory for preprocessed brain-extracted FLAIR images")
    parser.add_argument('--voxel_size', type=float, default=1.0, help="Resample voxel size (in mm)")
    parser.add_argument('--bet_frac', type=float, default=0.5, help="BET fractional intensity for brain extraction")
    args = parser.parse_args()
    return args

def normalize_intensity(img_data):
    """Z-score normalization"""
    mask = img_data > 0
    mean = img_data[mask].mean()
    std = img_data[mask].std()
    if std > 0:
        img_data[mask] = (img_data[mask] - mean) / std
    return img_data

def resample_to_voxel(infile, voxel_size, outfile):
    """Resample using FSL flirt to new voxel size"""
    os.system(f"flirt -in {infile} -ref {infile} -applyisoxfm {voxel_size} -out {outfile}")

def brain_extraction(infile, outfile, frac=0.5):
    os.system(f"bet {infile} {outfile} -f {frac} -g 0 -m")

def preprocess_flair(file_path, template, voxel_size, bet_frac, out_dir):
    # Keep original filename but force .nii.gz
    filename = file_path.stem.replace(".nii", "") + ".nii.gz"
    brain_file = out_dir / filename

    # Temporary working files
    tmp_reg = out_dir / f"tmp_reg_{filename}"
    tmp_res = out_dir / f"tmp_res_{filename}"

    # --- Registration ---
    os.system(f"flirt -in {file_path} -ref {template} -out {tmp_reg} -omat {out_dir}/{filename}_mat.mat")

    # --- Normalize intensity ---
    img = nib.load(tmp_reg)
    data = img.get_fdata()
    data = normalize_intensity(data)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), tmp_reg)  # overwrite

    # --- Resample voxel size ---
    resample_to_voxel(tmp_reg, voxel_size, tmp_res)

    # --- Brain extraction (final output) ---
    brain_extraction(tmp_res, brain_file, bet_frac)

    # --- Cleanup temp files ---
    os.remove(tmp_reg)
    os.remove(tmp_res)

    return brain_file

if __name__ == "__main__":
    args = parse()
    images_dir = args.dataset_dir / "images"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for f in os.listdir(images_dir):
        if "flair" in f.lower() and f.endswith((".nii", ".nii.gz")):
            file_path = images_dir / f
            print(f"Processing FLAIR: {file_path}")
            output_file = preprocess_flair(file_path, args.ref_template, args.voxel_size, args.bet_frac, args.output_dir)
            print(f"Saved: {output_file}")
