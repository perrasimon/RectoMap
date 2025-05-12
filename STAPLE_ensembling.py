import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from glob import glob

# Get base input directory from command-line (optional)
import sys
if len(sys.argv) < 2:
    print("Usage: python STAPLE_ensembling.py <base_predictions_folder>")
    sys.exit(1)

base_input_dir = sys.argv[1]

input_dirs = sorted(glob(os.path.join(base_input_dir, "predictions_fold*")))
output_dir = os.path.join(base_input_dir, "RectoMap_output")
os.makedirs(output_dir, exist_ok=True)

print("Found prediction folders:")
for d in input_dirs:
    print(f" - {d}")

# Get the common IDs across all folders
common_ids = None
for input_dir in input_dirs:
    model_ids = set(f for f in os.listdir(input_dir) if f.endswith('.nii.gz'))
    if common_ids is None:
        common_ids = model_ids
    else:
        common_ids &= model_ids

# Apply the STAPLE method for each ID
for file_id in common_ids:
    seg_stack = []
    
    # Load predictions from all models for the current subject
    for input_dir in input_dirs:
        seg_path = os.path.join(input_dir, file_id)
        seg_data = nib.load(seg_path).get_fdata().astype(np.uint64)
        seg_sitk = sitk.GetImageFromArray(seg_data)
        seg_stack.append(seg_sitk)

    # Apply the STAPLE method
    staple_seg_sitk = sitk.MultiLabelSTAPLE(seg_stack)

    # Convert the result back to a numpy array
    staple_seg = sitk.GetArrayFromImage(staple_seg_sitk).astype(np.int16)

    # Save the resulting mask
    affine = nib.load(os.path.join(input_dirs[0], file_id)).affine  # Use the affine from the first model
    staple_seg_img = nib.Nifti1Image(staple_seg, affine)
    output_path = os.path.join(output_dir, file_id)
    nib.save(staple_seg_img, output_path)

    print(f"Processed {file_id} and saved to {output_path}")

print("STAPLE ensemble process completed.")