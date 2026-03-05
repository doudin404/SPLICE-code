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
## 🚀 Getting Started

### 1. Data Preparation

Before training, you need to download the required dataset and set up the preprocessing tools.

* Download the PartNet dataset (`data_v0`).
* Download and install the `ManifoldPlus` executable.
* Update the configuration file `config/data/real.yaml` with your local absolute or relative paths:

```yaml
partnet_path: /mnt/d/data_v0/data_v0/      # Path to the downloaded PartNet dataset
manifold_plus_path: /mnt/d/data_v0/data_v0/ManifoldPlus.exe  # Path to the executable
output_dir: /mnt/d/data_v0/data_v0_sf/     # Destination folder for preprocessed data
```
* Run the data preprocessing script. The processed geometry data will be saved directly to your specified `output_dir`

``` Bash
python data_prepare.py
```
### 2. Training
The training pipeline is divided into two stages: training the main network and training the diffusion model.

**Phase 1: Main Model Training**
* Open `config/data/real.yaml` and update the data path to point to your preprocessed output directory:
```yaml
data_path: /mnt/d/data_v0/data_v0_sf/
```
* Run the primary training script. You can adjust other model hyperparameters directly in the YAML file. The checkpoint saving location and filename format can be configured via `ModelCheckpoint` within the training script.
``` Bash
python train_fry.py
```
### Download Pre-trained Models

We provide pre-trained weights hosted on Hugging Face. You can easily download them directly into the `checkpoints` directory using the official CLI:

1. Install the Hugging Face Hub library:
```bash
pip install -U "huggingface_hub[cli]"
```
2. Download the weights to the local ./checkpoints folder:
```bash
huggingface-cli download doudin404/splice --local-dir ./checkpoints --local-dir-use-symlinks False
```
### 3. Inference & Demo
To run the interactive demo, you can either use your newly trained weights or download our pre-trained checkpoints from [Insert Link Here].
* Update the configuration file `config/ui/real.yaml` to point to the correct checkpoint paths:
```yaml
model_path: ./checkpoints/splice.ckpt
adjust_path: ./checkpoints/fry/all_data.ckpt
pose_path: ./checkpoints/fry/all_pose.ckpt
```
* Launch the demo interface:
``` Bash
python demo.py
```

---
## 📑 Citation
If you find this work or code useful for your research, please cite our paper:
```
@article{zhou2025splice,
  title={SPLICE: Part-Level 3D Shape Editing from Local Semantic Extraction to Global Neural Mixing},
  author={Zhou, Jin and Yang, Hongliang and Xu, Pengfei and Huang, Hui},
  journal={arXiv preprint arXiv:2512.04514},
  year={2025}
}
```
## 📄 License
CC BY-NC 4.0