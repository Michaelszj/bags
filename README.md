# BAGS: Building Animatable Gaussian Splatting from a Monocular Video with Diffusion Priors
## Install
```bash
conda create -n bags python=3.10 -y && conda activate bags
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23 --no-deps --index-url https://download.pytorch.org/whl/cu118
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu118_pyt210.tar.bz2
pip install -r requirements.txt
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
pip install -e third_party/kmeans_pytorch

```
## Demo
We provide a demo on reconstructing an animal from a single monocualr video:
```bash
bash scripts/template.sh 0 camel "no" "no"
```
The results will be shown in ./logdir

## Acknowledgement
The codebase is from [BANMo](https://github.com/facebookresearch/banmo), and the SDS part is adapted from [DreamGaussian4d](https://github.com/jiawei-ren/dreamgaussian4d). We thank the authors for their brilliant works.

## Citation
```bash
@misc{zhang2024bagsbuildinganimatablegaussian,
      title={BAGS: Building Animatable Gaussian Splatting from a Monocular Video with Diffusion Priors}, 
      author={Tingyang Zhang and Qingzhe Gao and Weiyu Li and Libin Liu and Baoquan Chen},
      year={2024},
      eprint={2403.11427},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.11427}, 
}
```
