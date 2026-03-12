# Mouse Brain Registration with VoxelMorph using TensorFlow (GPU)

## Overview

This project implements **VoxelMorph**, a deep learning-based deformable image registration framework, to align 3D mouse brain T1-weighted MRI scans to a fixed template. All subject scans are first affinely aligned to the template, then refined using a trained neural network for high-precision, spatially-smooth deformations.

**Key Features:**
- GPU-accelerated training with NVIDIA TensorFlow
- Anatomically plausible deformations (validated via Jacobian analysis)
- Multiple similarity loss metrics (MSE, NCC, SSIM)
- Fast inference (~1 second per subject)
- Qualitative evaluation tools (warped images, displacement fields, deformation grids)

---

## System Requirements

| Component | Specification |
|-----------|---------------|
| Python | 3.9.13 |
| GPU | NVIDIA RTX 4070 (or compatible CUDA-capable GPU) |
| CUDA | 11.2+ |
| cuDNN | 8.1+ |
| TensorFlow | 2.10.0 (GPU-enabled) |
| NumPy | 1.24.3 |

**⚠️ Important:** Do not use NumPy ≥ 2.0 — it breaks TensorFlow compatibility.

---

## Installation

### Step 1: Create Virtual Environment

Open **Command Prompt (CMD)** and navigate to your project directory, then run:

```bash
"C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python39\python.exe" -m venv venv
```

Replace `<YourUsername>` with your actual Windows username.

### Step 2: Activate Virtual Environment

```bash
venv\Scripts\activate.bat
```

Verify the Python version:
```bash
python --version
```

Expected output: `Python 3.9.13`

### Step 3: Configure VS Code Interpreter (Optional)

In VS Code, select the interpreter: `C:\Users\<YourUsername>\Documents\<ProjectPath>\venv\Scripts\python.exe`

Verify with:
```python
import sys
print("Python:", sys.executable)
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Restart the kernel/terminal after installation.

### Step 5: Verify GPU Setup

Run the following in Python to confirm TensorFlow can access your GPU:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Expected output:**
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### GPU Troubleshooting

If GPU is not detected:

1. **Check CUDA files:** Ensure `cudart64_110.dll` and `cudnn64_8.dll` are in your CUDA `bin/` folder
2. **Verify CUDA installation:** Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. **Verify cuDNN:** Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) and extract to CUDA directory
4. **Update PATH:** Add CUDA `bin/` to your system PATH environment variable

---

## Project Structure

```
project/
├── venv/                           # Virtual environment
├── data/
│   ├── subject01.nii.gz           # Mouse brain T1-weighted MRI scans
│   ├── subject02.nii.gz
│   ├── ...
│   ├── subject30.nii.gz
│   └── warped_template.nii.gz     # Fixed template for alignment
├── bias_correction.py             # Correct field inhomogeneities in MRI scans
├── affine_registration.py         # Perform affine pre-alignment to template
├── Preprocessing.py               # Data loading and preprocessing utilities
├── trainandtest.py                # Main training and testing script
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

---

## Dataset Preparation

### Input Data Format

- **File format:** NIfTI (.nii.gz)
- **Image type:** 3D T1-weighted mouse brain MRI
- **Placement:** All subject scans should be placed in the `data/` folder

### Preprocessing Steps

The pipeline applies the following preprocessing in order:

1. **Bias Correction** (`bias_correction.py`): Corrects radiofrequency field inhomogeneities
2. **Affine Registration** (`affine_registration.py`): Aligns subjects to the template using 12-parameter affine transform
3. **Data Normalization** (`Preprocessing.py`): Normalizes intensity and prepares data for neural network training

---

## Usage

### Step 1: Bias Correction

```bash
python bias_correction.py
```

This script corrects low-frequency intensity variations in the MRI scans.

### Step 2: Affine Registration

```bash
python affine_registration.py
```

Performs rigid + affine alignment of all subject scans to `warped_template.nii.gz`.

### Step 3: Training and Testing

```bash
python trainandtest.py
```

This main script:
- Loads preprocessed data
- Trains the VoxelMorph neural network (60 epochs)
- Tests on validation set
- Outputs trained model and evaluation metrics

---

## Performance

| Metric | Value |
|--------|-------|
| Training Time (60 epochs) | ~20 minutes |
| Testing Time (per subject) | ~1.0 second |
| Similarity Loss (Best Model) | Lowest MSE among all runs |
| Anatomical Plausibility | Jacobian % ≤ 0: **0.01%** (excellent) |

**Note:** Jacobian determinant < 0 indicates local folding/compression. Our model maintains 99.99% anatomically plausible deformations.

---

## Evaluation Metrics

The project uses the following metrics to assess registration quality:

### Similarity Losses

- **Mean Squared Error (MSE):** Measures pixel-wise intensity differences between warped and template images
- **Normalized Cross-Correlation (NCC):** Correlation-based similarity (robust to intensity scaling)
- **Structural Similarity Index (SSIM):** Perceptual similarity metric (combines luminance, contrast, structure)

### Anatomical Plausibility

- **Jacobian Determinant:** Validates that deformations are physically realistic
  - `det(J) > 0`: Local expansion or rotation (valid)
  - `det(J) ≤ 0`: Local folding or compression (anatomically implausible)
  - Target: < 1% of voxels with `det(J) ≤ 0`

### Qualitative Assessment

Generated outputs for visual inspection:
- **Warped Images:** Subject scans transformed to template space
- **Displacement Fields:** 3D vector maps showing deformation magnitude and direction
- **Deformation Grids:** Wireframe visualization of spatial deformation

---

## Output Files

After running `trainandtest.py`, the following files are generated:

- `trained_model/` — Saved neural network weights and architecture
- `warped_subjects/` — Affinely + deformably aligned subject images
- `displacement_fields/` — 3D displacement vector maps
- `deformation_grids/` — Visualizations of spatial transformation
- `evaluation_metrics.csv` — Quantitative metrics (loss, Jacobian statistics, etc.)

---

## Requirements File

The `requirements.txt` file contains all necessary Python packages. View it to see exact versions:

```
tensorflow==2.10.0
numpy==1.24.3
nibabel>=3.0.0
scipy>=1.8.0
scikit-image>=0.18.0
```

To update dependencies in the future:

```bash
pip install --upgrade -r requirements.txt
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install -r requirements.txt` and restart kernel |
| `CUDA_ERROR_NOT_INITIALIZED` | Ensure GPU is detected via `tf.config.list_physical_devices('GPU')` |
| `NumPy version conflict` | Use NumPy 1.24.3: `pip install numpy==1.24.3` |
| Out of GPU memory | Reduce batch size in `trainandtest.py` or use a smaller training subset |
| MRI files not found | Check that `.nii.gz` files are in `data/` folder and filenames match the script |

### Getting Help

1. Check Python version: `python --version`
2. Verify GPU status: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
3. Check CUDA/cuDNN installation: `nvidia-smi` (shows GPU info) and verify DLLs in CUDA bin folder

---

## References

- **VoxelMorph Paper:** [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/abs/1809.05231)
- **TensorFlow Documentation:** https://www.tensorflow.org/
- **NIfTI Format:** [Neuroimaging Informatics Technology Initiative](https://nifti.nimh.nih.gov/)

---

## License

This project is provided for research and educational purposes.

---

## Author

**Authors:** Lina Cheung, Giuliana Fagre Guerriero, Vivian Lu
**Institution:** Columbia University Biomedical Engineering

---

## Notes

- All mouse brain scans must be **pre-aligned affinely** to the template before VoxelMorph deformable registration
- The VoxelMorph model is trained to output **spatially smooth deformations** via diffeomorphism constraints
- Jacobian validation ensures deformations are physiologically realistic
- GPU acceleration is strongly recommended; CPU-only training is feasible but significantly slower