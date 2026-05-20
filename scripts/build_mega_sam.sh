#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/../third_party/mega_sam/base"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: Please activate the conda environment first, for example: conda activate pdi-bench"
    exit 1
fi

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Install CUDA 11.8 toolkit with conda before building."
    exit 1
fi

CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
if [ "$CUDA_VERSION" != "11.8" ]; then
    echo "ERROR: Expected CUDA 11.8 from the conda environment, but nvcc reports CUDA $CUDA_VERSION"
    echo "CUDA_HOME=$CUDA_HOME"
    echo "nvcc=$(command -v nvcc)"
    exit 1
fi

TORCH_CUDA_VERSION="$(python -c "import torch; print(torch.version.cuda)")"
if [ "$TORCH_CUDA_VERSION" != "11.8" ]; then
    echo "ERROR: Expected PyTorch cu118, but torch.version.cuda is $TORCH_CUDA_VERSION"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: $BASE_DIR not found. Did you run: git submodule update --init --recursive ?"
    exit 1
fi

cd "$BASE_DIR"

SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")

echo "=== Step 1/4: Building droid_backends ==="
cat > setup.py << 'SETUP_EOF'
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension(
            'droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
SETUP_EOF

pip install -e . --no-build-isolation

echo "=== Step 2/4: Copying droid_backends.so to site-packages ==="
cp droid_backends*.so "$SITE_PKG/"

echo "=== Step 3/4: Building lietorch ==="
cat > setup.py << 'SETUP_EOF'
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension(
            'lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen'),
            ],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp',
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
SETUP_EOF

pip install -e . --no-build-isolation

echo "=== Step 4/4: Copying lietorch_backends.so to site-packages ==="
cp thirdparty/lietorch/lietorch_backends*.so "$SITE_PKG/"

echo "=== Restoring setup.py ==="
git checkout setup.py

echo ""
echo "=== Verifying ==="
python -c "import droid_backends; print('droid_backends OK')"
python -c "from lietorch import SE3; p = SE3.Identity(1, device='cuda'); p.inv(); print('lietorch OK')"
echo "All done."
