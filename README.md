<h1>
  <img src="https://github.com/user-attachments/assets/42afc94e-ce51-499b-9033-bb5b4864e886" alt="MRI Icon" width="40"/>
  Rectal Tumor and Mesorectum Segmentation in MRI
</h1>


This repository provides a deep learning model for the segmentation of **rectal cancer** and **mesorectum** from T2-weighted MRI images, using a [Swin-UMamba-based architecture].

## 🧪 Model Details
- Framework: PyTorch + MONAI + nnUNet
- Trained on: [brief description of your dataset]
- Targets: Label 1 = rectal tumor, Label 2 = mesorectum

## 📦 Installation

```bash
conda create -n rectal-seg python=3.9
conda activate rectal-seg
pip install -r requirements.txt
