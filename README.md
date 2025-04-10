<h1>
  <img src="[https://cdn-icons-png.flaticon.com/512/5048/5048368.png](https://github.com/user-attachments/assets/9126b0d9-af0b-4141-8336-590c9a7b2f53)" alt="MRI Icon" width="40"/>
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
