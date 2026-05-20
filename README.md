# PDI-Bench: Perspective Distortion Index for AI Video World Models

**PDI-Bench** is an automated evaluation framework designed to quantify **spatial scale and perspective consistency** in AI video generation models (such as Sora, Seedance, Flow). By integrating **SAM2**, **Co-Tracker**, and **Mega-SAM**, this project builds a physical-audit pipeline from 2D pixel tracking to 3D geometric reconstruction.

![Demo Preview](figures/bus_hero.gif)

![Pipeline](figures/pipeline.png)

---

## Core Evaluation Logic

### 1. Scale-Depth Alignment (Spatial Dimension, $\epsilon_{scale}$)
- **Core principle**: This term is grounded in the pinhole camera model. In the physical world, an object's **pixel height ($h$) multiplied by its physical depth ($Z$) remains constant** (i.e., $h \cdot Z = f \cdot H$).
- **What it audits**: It measures whether object scale changes during forward/backward motion strictly follow perspective geometry.
- **Hallucinations it captures**: Perspective inconsistency artifacts frequently seen in AI videos, such as "the object moves away but does not shrink" (giant-like drift) or "the object does not move yet suddenly shrinks" (volume collapse).

### 2. Motion Consistency (Temporal Dimension, $\epsilon_{traj}$)
- **Core principle**: This term is based on Newtonian motion (inertia). For macroscopic objects, trajectories in 3D space should be continuous and smooth, with **no abrupt acceleration jumps** and **no unjustified directional reversals**.
- **What it audits**: It directly analyzes centroid motion vectors in 3D world coordinates, quantifying both acceleration discontinuity (magnitude) and turning behavior (directional angle change).
- **Hallucinations it captures**: It is robust to camera shake and specifically detects non-inertial artifacts in AI videos, including high-frequency jitter, instantaneous teleportation, and momentum-violating sharp turn-backs.

### 3. Structural Rigidity (Material Dimension, $\epsilon_{rigidity}$)
- **Core principle**: This term is based on rigid-body invariance. In the physical world, the **3D distance between any two points inside a rigid object should remain constant over time**.
- **What it audits**: Using dense point tracking (CoTracker), it samples multiple 3D anchor pairs within the object and monitors whether their distance ratios remain stable throughout motion.
- **Hallucinations it captures**: It targets the notorious **Jello Effect** in AI videos, detecting local melting, non-physical deformation, and stretching artifacts during motion (e.g., elongated car fronts or warped faces).

The **Perspective Distortion Index (PDI)** is defined as a weighted sum of three orthogonal residuals:

$$
\text{PDI} = w_1 \cdot \mathrm{RMSE}(\epsilon_{scale}) + w_2 \cdot \mathrm{RMSE}(\epsilon_{traj}) + w_3 \cdot \epsilon_{rigidity}
$$

where $\sum_{i=1}^{3} w_i = 1$. Each component is designed to be scale-invariant and to capture a geometrically orthogonal failure mode.

---

## 1. Environment Requirements

This project is highly sensitive to CUDA versions. **You must strictly follow the version combination below**:

- **Python**: 3.10
- **CUDA Toolkit**: 11.8
- **PyTorch**: 2.1.0
- **Conda environment name**: `pdi-bench`

Do **not** rely on a system-wide CUDA installation such as `/usr/local/cuda-12.x` or `/usr/local/cuda-13.x`. PDI-Bench should use the CUDA 11.8 toolkit installed inside the conda environment. If your shell startup file (`~/.bashrc`, `~/.zshrc`, etc.) contains a line like the following, remove it or comment it out before continuing:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
```

Also do **not** create the environment from `third_party/mega_sam/environment.yml` or install `third_party/mega_sam/UniDepth/requirements.txt` directly. Those upstream files pin different PyTorch/CUDA versions and can overwrite the version combination above.

---

## 2. Clone the Project and Submodules

This project includes nested submodules: `third_party/mega_sam` itself depends on `third_party/mega_sam/base` (the DROID-SLAM core).

```bash
git clone --recursive https://github.com/AnteaWu/PDI-Bench.git
cd PDI-Bench

# If the main repo is already cloned, initialize submodules recursively (including nested ones)
git submodule update --init --recursive
```

---

## 3. Environment Setup

### 3.1 Create a Conda Environment

```bash
conda create -n pdi-bench python=3.10 -y
conda activate pdi-bench

# Install basic build tools
conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11 cmake -y

# Install PyTorch (you must specify `index-url`)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install the CUDA 11.8 build toolkit inside this conda environment
conda install -c nvidia cuda-nvcc=11.8 cuda-cccl=11.8 cuda-libraries-dev=11.8 cuda-cudart-dev=11.8 libcublas-dev=11.11 -y
```

### 3.2 Configure CUDA for This Conda Environment

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/pdi_bench_cuda.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
EOF

conda deactivate
conda activate pdi-bench
```

Verify that PyTorch and `nvcc` both use CUDA 11.8 from the active conda environment:

```bash
python -c "import os, torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA_HOME:', os.environ.get('CUDA_HOME'))"
which nvcc
nvcc --version
```

Expected results:

- `torch CUDA` should be `11.8`.
- `CUDA_HOME` should point to the active conda environment, not `/usr/local/cuda-*`.
- `which nvcc` should point to `$CONDA_PREFIX/bin/nvcc`.

---

## 4. Install Dependencies

### 4.1 Install Basic Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `torch-scatter` and `SAM2`/`Co-Tracker` are **not** in `requirements.txt` and must be installed separately in sections 4.2 and 4.3 below.

Install the additional runtime packages used by Mega-SAM and UniDepth without changing the pinned PyTorch/CUDA stack:

```bash
pip install wandb yacs h5py safetensors tabulate
pip install xformers==0.0.22.post7 --no-deps
```

### 4.2 Install SAM2 and Co-Tracker

```bash
pip install hydra-core iopath
pip install --no-deps git+https://github.com/facebookresearch/segment-anything-2.git
pip install --no-deps git+https://github.com/facebookresearch/co-tracker.git
```

> **Important**: `--no-deps` prevents these packages from upgrading the pinned PyTorch/CUDA stack. Do not follow external install instructions that install or upgrade CUDA packages for SAM2 or Co-Tracker.

### 4.3 Install `torch-scatter` (must force the pt21 build)

> **Important**: Running `pip install torch-scatter` directly may install an older pt20 build and cause `undefined symbol` runtime errors. You must use `--force-reinstall` to ensure the version matches PyTorch 2.1.0.

```bash
pip install torch-scatter --force-reinstall -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

Verify installation:
```bash
python -c "from torch_scatter import scatter_sum; print('torch_scatter OK')"
```

Verify the main runtime packages:

```bash
PYTHONPATH=third_party/mega_sam/UniDepth python -c "import sam2, xformers; from cotracker.predictor import CoTrackerPredictor; from unidepth.models import UniDepthV2; print('SAM2 / CoTracker / UniDepth OK')"
```

### 4.4 Compile Mega-SAM Low-Level Operators

The DROID-SLAM core of Mega-SAM depends on two CUDA C++ extensions: `droid_backends` and `lietorch`. Run the provided build script from the **project root**:

```bash
conda activate pdi-bench
bash scripts/build_mega_sam.sh
```

The script will build and install both extensions, copy the compiled `.so` files into site-packages, and restore `setup.py` automatically. Upon success you will see:

```
droid_backends OK
lietorch OK
All done.
```

> **Note**: It is normal to see many warnings such as `-Wdeprecated-declarations` and `-Wreorder` during compilation. They do not affect usage. Only lines with `error:` require action.

---

## 5. Download Model Weights

Download the following checkpoint files into the corresponding directories:

### SAM2
```bash
mkdir -p checkpoints/sam2
wget -P checkpoints/sam2 https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Copy the config file from the installed SAM2 package
python -c "import shutil, sam2; import os; shutil.copy(os.path.join(os.path.dirname(sam2.__file__), 'configs/sam2/sam2_hiera_l.yaml'), 'checkpoints/sam2/sam2_hiera_l.yaml')"
```

### Co-Tracker (CoTracker3 Offline)
```bash
mkdir -p checkpoints/tracker
wget -P checkpoints/tracker https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

### Mega-SAM: Depth-Anything
```bash
mkdir -p third_party/mega_sam/Depth-Anything/checkpoints
wget -P third_party/mega_sam/Depth-Anything/checkpoints \
  https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
```

### Mega-SAM: megasam_final.pth
```bash
mkdir -p third_party/mega_sam/checkpoints
# Get this file from the official Mega-SAM repository: https://github.com/mega-sam/mega-sam
```

After downloading, the file must exist at:

```bash
test -f third_party/mega_sam/checkpoints/megasam_final.pth && echo "megasam_final.pth OK"
```

### Mega-SAM: RAFT (required for CVD-consistent depth optimization)

> RAFT is required in Step 4 of the full MegaSAM pipeline (CVD pre-flow). If missing, the pipeline will automatically fall back to raw DROID depth, but temporal depth consistency will degrade.

```bash
pip install gdown
cd third_party/mega_sam/cvd_opt/
gdown 1R8m_jMvCun-N45XkMvHlG0P38kXy-h6I
cd ../../../
```

Weight paths are configured in `configs/default.yaml` and can be edited as needed.

Verify all required checkpoint files:

```bash
test -f checkpoints/sam2/sam2_hiera_large.pt && test -f checkpoints/sam2/sam2_hiera_l.yaml && test -f checkpoints/tracker/scaled_offline.pth && test -f third_party/mega_sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth && test -f third_party/mega_sam/checkpoints/megasam_final.pth && echo "Required checkpoints OK"
```

---

## 6. Download Dataset

The benchmark videos are hosted on Hugging Face: [AnteaWu/PDI-Dataset](https://huggingface.co/datasets/AnteaWu/PDI-Dataset).

### Install Hugging Face Hub CLI

```bash
pip install huggingface_hub
```

### Download Ground Truth Videos Only (Recommended)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='AnteaWu/PDI-Dataset',
    repo_type='dataset',
    allow_patterns='GT/**',
    local_dir='videos'
)
"
```

After downloading, the GT videos will be placed under `videos/GT/`, structured as:

```
videos/
  GT/
    Biological_Motion/
    Curved_Motion/
    Dynamic_Tracking/
    Longitudinal_Convergence/
    Partial_Occlusion/
```

### Download All Videos (GT + Generated)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='AnteaWu/PDI-Dataset',
    repo_type='dataset',
    local_dir='videos'
)
"
```

---

## 7. Quick Start

### Specify Target by Text (recommended, fully automatic)

```bash
conda activate pdi-bench
python evaluation/main.py --input your_video.mp4 --text "train"
```

### Specify Target with Manual Coordinates

```bash
conda activate pdi-bench
python evaluation/main.py --input your_video.mp4 --points '[[500, 500]]'
```

### Full Argument Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input` | Required | Input video path |
| `--text` | None | Text description of the target object, auto-localized with Florence-2 |
| `--points` | None | Manual click coordinates in format `[[x, y]]`, mutually exclusive with `--text` |
| `--config` | `configs/default.yaml` | Configuration file path |
| `--output_dir` | `results` | Output directory |

---
