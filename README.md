<<h1 align="center">>
  <img src="https://github.com/user-attachments/assets/42afc94e-ce51-499b-9033-bb5b4864e886" alt="Icon" width="50"/>
  Rectal Tumor and Mesorectum Segmentation in MRI
</h1>


🔬 About This Repository
This repository provides a deep learning pipeline for the segmentation of rectal cancer and mesorectum from T2-weighted MRI scans.
The system integrates five different deep learning models:

nnUNet

uMambaBot

Swin-UMamba (pretrained)

Swin-UMamba (from scratch)

Swin-UNETR

The predictions from each model are combined using a STAPLE-based ensemble approach, enhancing the robustness and accuracy of the final segmentation.

## 🧪 Model Details
- Framework: PyTorch + MONAI + nnUNet
- Trained on: [brief description of your dataset]
- Targets: Label 1 = rectal tumor, Label 2 = mesorectum

## 📦 Installation

```bash
conda create -n rectal-seg python=3.9
conda activate rectal-seg
pip install -r requirements.txt
