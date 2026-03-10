Official implementation of SPLICE: a part-level 3D shape editing framework based on local semantic extraction and global neural mixing.
---
# SPLICE: Part-Level 3D Shape Editing from Local Semantic Extraction to Global Neural Mixing

[![arXiv](https://img.shields.io/badge/arXiv-2512.04514-b31b1b.svg)](https://arxiv.org/abs/2512.04514)
[![Conference](https://img.shields.io/badge/Accepted-Pacific_Graphics_2025-blue.svg)]()
![](assets/teaser.png)
> **SPLICE: Part-Level 3D Shape Editing from Local Semantic Extraction to Global Neural Mixing** <br>
> Jin Zhou, Hongliang Yang, Pengfei Xu, Hui Huang <br>
> *Pacific Graphics 2025*



## Abstract

Neural implicit representations of 3D shapes have shown great potential in 3D shape editing due to their ability to model high-level semantics and continuous geometric representations. However, existing methods often suffer from limited editability, lack of part-level control, and unnatural results when modifying or rearranging shape parts. 

In this work, we present **SPLICE**, a novel part-level neural implicit representation of 3D shapes that enables intuitive, structure-aware, and high-fidelity shape editing. By encoding each shape part independently and positioning them using parameterized Gaussian ellipsoids, SPLICE effectively isolates part-specific features while discarding global context that may hinder flexible manipulation. A global attention-based decoder is then employed to integrate parts coherently, further enhanced by an attention-guiding filtering mechanism that prevents information leakage across symmetric or adjacent components. 

## Key Capabilities

Through this architecture, SPLICE supports a wide range of intuitive part-level editing operations while preserving semantic consistency and structural plausibility:
* **Translation, Rotation, & Scaling**
* **Part Deletion & Duplication**
* **Cross-Shape Part Mixing**
---

# 🚀 Getting Started

## Environment Configuration

Ensure you have Python installed, then run the following commands to set up your environment. This project uses PyTorch 2.6.0 with CUDA 12.6.

```bash
# Core PyTorch installation
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Sparse convolution and PyTorch Lightning
pip install spconv-cu120
python -m pip install lightning

# Torch Scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# 3D processing, utilities, and logging
pip install vtk==9.3.1
pip install timm einops addict python-dotenv
pip install hydra-core omegaconf trimesh h5py scipy 
pip install pyvista scikit-image libigl wandb pynput

# Flash Attention (requires packaging and ninja first)
pip install packaging ninja
pip install flash-attn --no-build-isolation

```

## Dataset Preparation

1. **Clone the Shape-As-Points repository** to get the dataset scripts:
```bash
git clone https://github.com/autonomousvision/shape_as_points.git
cd shape_as_points
bash scripts/download_shapenet.sh

```


2. **Download the datasets** and organize them into your `data/` directory:
* Place the downloaded `shapenet_psr` dataset into `data/`.
* Download the `PartNet` dataset and place it into `data/`.
* Ensure `partnet_metadata.json` is located directly in the `data/` folder (or inside `data/shapenet/` depending on your specific code config).



**Expected Directory Structure:**

```text
data/
├── partnet_metadata.json
├── shapenet/                  # From shapenet_psr
│   ├── {shapenet_id_1}/
│   │   └── psr.npz
│   └── ...
└── {partnet_id_1}/            # PartNet samples
    ├── point_sample/
    │   ├── pts-10000.txt
    │   └── label-10000.txt
    └── ...

```

## 🏋️ Training Pipeline

The training process is divided into three sequential stages:

1. **Stage 1: Base Model Training**
Run the main training script. This will generate the initial model checkpoints.
```bash
python train.py

```


*Output:* Checkpoints will be saved in the `checkpoints_splice/` directory.
2. **Stage 2: Generate Diffusion Training Data**
Extract the features/data required to train the diffusion generative model.
```bash
python pred_diff.py

```


*Output:* Training data is exported to `export/prediff/`.
3. **Stage 3: Diffusion Model Training**
Train the diffusion model using the data generated in the previous step.
```bash
python traindiff.py

```


*Output:* Diffusion checkpoints will be saved in `checkpoints_splice_diff/`.

## 📥 Pre-trained Models (Optional)

If you want to skip training, you can download pre-trained weights directly from Hugging Face:

```bash
huggingface-cli download doudin404/splice --local-dir ./checkpoints --local-dir-use-symlinks False

```

## 🧪 Testing & Inference

Once you have trained or downloaded the checkpoints, you can evaluate the models using the following scripts:

* **View Reconstruction Results:**
```bash
python pred.py

```


* **View Random Generation Results:**
```bash
python samplediff.py

```


* **Interactive Demo:**
Launch the interactive GUI window. (Note: Operation methods and controls are similar to the [SPAGHETTI](https://github.com/amirhertz/spaghetti) project).
```bash
python demo.py

```



---

Would you like me to add a section explaining the arguments for the `demo.py` or `train.py` scripts, or is this README good to go for your repository?