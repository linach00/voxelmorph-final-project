# Mouse Brain Registration with VoxelMorph using TensorFlow (GPU)

## Setup Summary

- **Python version**: 3.9.13
- **Virtual env path**: `venv/`
- **GPU**: NVIDIA RTX 4070
- **CUDA version**: 11.2
- **cuDNN version**: 8.1
- **TensorFlow version**: 2.10.0 (GPU-enabled)
- **NumPy version**: 1.24.3 (for compatibility)

---

## Setup Instructions

```
In CMD : 
Create the environemnt with python 3.9 version 

example:
"C:\Users\lcheu\AppData\Local\Programs\Python\Python39\python.exe" -m venv venv

Activate the environment
venv\Scripts\activate.bat

Check if python version is correct 
python --version
should be: Python 3.9.13

In VS code, sleect interpreter: C:\Users\lcheu\Documents\BMEN4460_Project\venv\Scripts\python.exe
confirm with : import sys, print("Python:", sys.executable)

2. pip install -r requirements.txt and restart kernel 

3. verify GPU setup with 
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

Output should look like: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

Notes:
Do not use NumPy >= 2.0 — breaks TensorFlow compatibility

Make sure you have cudart64_110.dll and cudnn64_8.dll in your CUDA bin/ folder to run tensorflow on GPU 

Project Structure 
project/
├── venv/
├── data/
│   ├── subject01.nii.gz
│   ├── subject##.nii.gz 
│   ├── warped_template.nii.gz
├── bias_correction.py
├── affine_registration.py
├── Preprocessing.py
├── trainandtest.py
├── requirements.txt
└── README.md

```

## Dataset and Template
This project uses 3D mouse brain T1-weighted MRIs. All subject scans are affinely aligned to a fixed template prior to VoxelMorph registration

## Performance 
Training time (60 epochs): ~20 minutes
Testing time per subject: ~1.0 sec
Best model (MSE loss)
Similarity Loss: Lowest among all
Jacobian % ≤ 0: 0.01% (high anatomical plausibility)

### Evaluation Metrics
Similarity Losses: MSE, NCC, SSIM
Anatomical Plausibility: % of voxels with non-positive Jacobian
Qualitative: Warped image, RGB displacement, deformation grid
