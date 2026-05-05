# PDI-Bench: Perspective Distortion Index for AI Video World Models

**PDI-Bench** is an automated evaluation framework designed to quantify **spatial scale and perspective consistency** in AI video generation models (such as Sora, Seedance, Flow). By integrating **SAM2**, **Co-Tracker**, and **Mega-SAM**, this project builds a physical-audit pipeline from 2D pixel tracking to 3D geometric reconstruction.

---

## Core Evaluation Logic

### 1. Scale-Depth Alignment (Spatial Dimension, $\epsilon_{scale}$)
- **Core principle**: This term is grounded in the pinhole camera model. In the physical world, an object's **pixel height ($h$) multiplied by its physical depth ($Z$) remains constant** (i.e., $h \cdot Z = f \cdot H$).
- **What it audits**: It measures whether object scale changes during forward/backward motion strictly follow perspective geometry.
- **Hallucinations it captures**: Perspective inconsistency artifacts frequently seen in AI videos, such as "the object moves away but does not shrink" (giant-like drift) or "the object does not move yet suddenly shrinks" (volume collapse).

### 2. Kinematic Consistency (Temporal Dimension, $\epsilon_{traj}$)
- **Core principle**: This term is based on Newtonian motion (inertia). For macroscopic objects, trajectories in 3D space should be continuous and smooth, with **no abrupt acceleration jumps** and **no unjustified directional reversals**.
- **What it audits**: It directly analyzes centroid motion vectors in 3D world coordinates, quantifying both acceleration discontinuity (magnitude) and turning behavior (directional angle change).
- **Hallucinations it captures**: It is robust to camera shake and specifically detects non-inertial artifacts in AI videos, including high-frequency jitter, instantaneous teleportation, and momentum-violating sharp turn-backs.

### 3. Structural Rigidity (Material Dimension, $\epsilon_{rigidity}$)
- **Core principle**: This term is based on rigid-body invariance. In the physical world, the **3D distance between any two points inside a rigid object should remain constant over time**.
- **What it audits**: Using dense point tracking (CoTracker), it samples multiple 3D anchor pairs within the object and monitors whether their distance ratios remain stable throughout motion.
- **Hallucinations it captures**: It targets the notorious **Jello Effect** in AI videos, detecting local melting, non-physical deformation, and stretching artifacts during motion (e.g., elongated car fronts or warped faces).

The **Perspective Distortion Index (PDI)** is defined as a weighted sum of three orthogonal residuals:

$$
\text{PDI} = w_1 \cdot \operatorname{RMSE}(\epsilon_{scale}) + w_2 \cdot \operatorname{RMSE}(\epsilon_{traj}) + w_3 \cdot \sigma_{rigidity}
$$

where $\sum_{i=1}^{3} w_i = 1$. Each component is designed to be scale-invariant and to capture a geometrically orthogonal failure mode.

---

## 1. Environment Requirements

This project is highly sensitive to CUDA versions. **You must strictly follow the version combination below**:

- **Python**: 3.10
- **CUDA Toolkit**: 11.8
- **PyTorch**: 2.1.0

---

## 2. Environment Setup

### 2.1 Create a Conda Environment

```bash
conda create -n pdi_eval python=3.10 -y
conda activate pdi_eval

# Install basic build tools
conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11 cmake -y

# Install PyTorch (you must specify `index-url`)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install CUDA toolkit and ensure `nvcc` matches cu118
conda install -c nvidia cuda-toolkit=11.8 -y
```

### 2.2 Set Environment Variables

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

> It is recommended to add the three lines above to `~/.bashrc` or `~/.zshrc` for persistence.

---

## 3. Clone the Project and Submodules

This project includes nested submodules: `third_party/mega_sam` itself depends on `third_party/mega_sam/base` (the DROID-SLAM core).

```bash
git clone --recursive https://github.com/JiaxinWu-25/PDI-Eval.git
cd PDI-Eval

# If the main repo is already cloned, initialize submodules recursively (including nested ones)
git submodule update --init --recursive
```

### 3.1 Apply the PyTorch 2.1 Compatibility Patch

`projective_ops.py` in `mega_sam/base` uses lietorch Lie-group operations. Under PyTorch 2.1, it may crash because the `AutocastCUDA` dispatch key is not recognized. Apply the following manual patch:

```bash
cd third_party/mega_sam/base/droid_slam/geom

python - <<'EOF'
import pathlib

f = pathlib.Path('projective_ops.py')
src = f.read_text()

old1 = (
    "  # transform\n"
    "  Gij = poses[:, jj] * poses[:, ii].inv()\n"
    "\n"
    "  ## WHAT HACK IS THIS LINE?\n"
    "  ## I think it's for stereo rig!\n"
    "  # Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=\"cuda\")\n"
    "  X1, Ja = actp(Gij, X0, jacobian=jacobian)"
)
new1 = (
    "  # transform - lietorch Lie group ops do not support AutocastCUDA dispatch key\n"
    "  with torch.cuda.amp.autocast(enabled=False):\n"
    "    Gij = poses[:, jj] * poses[:, ii].inv()\n"
    "    X1, Ja = actp(Gij, X0, jacobian=jacobian)\n"
    "\n"
    "  ## WHAT HACK IS THIS LINE?\n"
    "  ## I think it's for stereo rig!\n"
    "  # Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=\"cuda\")"
)

if old1 in src:
    src = src.replace(old1, new1)
    f.write_text(src)
    print("patch applied to projective_transform")
else:
    print("projective_transform: already patched or source changed, skipping")
EOF

cd ../../../../
```

---

## 4. Install Dependencies

### 4.1 Install Basic Python Dependencies

```bash
pip install -r requirements.txt
```

### 4.2 Install SAM2 and Co-Tracker

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pip install git+https://github.com/facebookresearch/co-tracker.git
```

### 4.3 Install `torch-scatter` (must force the pt21 build)

> **Important**: Running `pip install torch-scatter` directly may install an older pt20 build and cause `undefined symbol` runtime errors. You must use `--force-reinstall` to ensure the version matches PyTorch 2.1.0.

```bash
pip install torch-scatter --force-reinstall -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

Verify installation:
```bash
python -c "from torch_scatter import scatter_sum; print('torch_scatter OK')"
```

### 4.4 Compile Mega-SAM Low-Level Operators

The DROID-SLAM core of Mega-SAM depends on two CUDA C++ extensions: `droid_backends` and `lietorch`. The compiled binaries must exactly match your current PyTorch version, otherwise you may get `undefined symbol` or `Unrecognized tensor type ID: AutocastCUDA` errors.

```bash
cd third_party/mega_sam/base

# Step 1: Build and install `droid_backends`
cp setup_droid.py setup.py
pip install -e . --no-build-isolation

# Step 2: Copy built `droid_backends.so` into site-packages
#         Python runtime loads from site-packages by default; without copying, an older binary may be loaded and cause ABI errors
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
cp droid_backends*.so "$SITE_PKG/"

# Step 3: Build and install `lietorch`
cp setup_lie.py setup.py
pip install -e . --no-build-isolation

# Step 4: Also copy built `lietorch_backends.so` into site-packages
#         Same reason: old binaries may miss AutocastCUDA dispatch-key registration and will crash under PyTorch 2.1
cp thirdparty/lietorch/lietorch_backends*.so "$SITE_PKG/"

# Step 5: Restore `setup.py`
cp setup_org.py setup.py

cd ../../../
```

Verify installation:
```bash
python -c "import droid_backends; print('droid_backends OK')"
python -c "from lietorch import SE3; p = SE3.Identity(1, device='cuda'); p.inv(); print('lietorch OK')"
```

> **Note**: It is normal to see many warnings such as `-Wdeprecated-declarations` and `-Wreorder` during compilation. They do not affect usage. Only lines with `error:` require action.

---

## 5. Download Model Weights

Download the following checkpoint files into the corresponding directories:

### SAM2
```bash
mkdir -p checkpoints/sam2
wget -P checkpoints/sam2 https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

> The `sam2_hiera_l.yaml` config file is included in the SAM2 package. The default path is `checkpoints/sam2/sam2_hiera_l.yaml`.

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

### Mega-SAM: RAFT (required for CVD-consistent depth optimization)

> RAFT is required in Step 4 of the full MegaSAM pipeline (CVD pre-flow). If missing, the pipeline will automatically fall back to raw DROID depth, but temporal depth consistency will degrade.

```bash
pip install gdown
cd third_party/mega_sam/cvd_opt/
gdown 1R8m_jMvCun-N45XkMvHlG0P38kXy-h6I
cd ../../../
```

Weight paths are configured in `configs/default.yaml` and can be edited as needed.

---

## 6. Quick Start

### Specify Target by Text (recommended, fully automatic)

```bash
python main.py --input data/your_video.mp4 --text "train"
```

### Specify Target with Manual Coordinates

```bash
python evaluation/main.py --input data/your_video.mp4 --text "your_video"
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
