## Installation

### Requirements
- Linux with Python ≥ 3.11
- PyTorch ≥ 2.6 and torchvision that matches the PyTorch installation (0.21.0).
- CUDA 12 was used during development.
- Detectron2: a modified Detectron 2 is required that contains several bugfixes that primarily show up on large supercomputer clusters running SLURM, available [here](https://github.com/mdsmith-cim/detectron2).
- OpenCV >=4.11
- A GPU with around 24GB of memory. Compatible with anything that PyTorch supports.

These requirements are listed in the requirements.txt file, so they should be installable with your favourite package manager such as pip or uv:
- `pip install -r requirements.txt`

### Example installation with conda forge
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh -b -p my_python_env
source my_python_env/bin/activate
conda install cuda==12.6 cuda-version==12.6
conda install h5py tqdm packaging scikit-image submitit pandas ipython scikit-learn ipython shapely scipy cython
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install timm pycocotools tensorboard oflibpytorch cityscapesscripts
# --no-build-isolation because detectron2 has packaging issues
pip install 'git+https://github.com/mdsmith-cim/detectron2.git' 'git+https://github.com/cocodataset/panopticapi.git' --no-build-isolation
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### CUDA kernel for MSDeformAttn
One requirement not listed is the custom CUDA kernel required by Mask2Former. Run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
