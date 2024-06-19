

<div align="center">

<samp>

<h2> LLCaps: Learning to Illuminate Low-Light Capsule Endoscopy with Curved Wavelet Attention and Reverse Diffusion </h1>

<h4> Long Bai *, Tong Chen *, Yanan Wu, An Wang, Mobarakol Islam, and Hongliang Ren </h3>

<h3> Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023 (Oral) </h2>

</samp>   

| **[[```arXiv```](<https://arxiv.org/abs/2307.02452>)]** |
|:-------------------:|

---

</div>     

If you find our code or paper useful, please cite as

```bibtex
@article{bai2023llcaps,
  title={LLCaps: Learning to Illuminate Low-Light Capsule Endoscopy with Curved Wavelet Attention and Reverse Diffusion},
  author={Bai, Long and Chen, Tong and Wu, Yanan and Wang, An and Islam, Mobarakol and Ren, Hongliang},
  journal={arXiv preprint arXiv:2307.02452},
  year={2023}
}
```
---
## Abstract

Wireless capsule endoscopy (WCE) is a painless and non-invasive diagnostic tool for gastrointestinal (GI) diseases. However, due to GI anatomical constraints and hardware manufacturing limitations, WCE vision signals may suffer from insufficient illumination, leading to a complicated screening and examination procedure. Deep learning-based low-light image enhancement (LLIE) in the medical field gradually attracts researchers. Given the exuberant development of the denoising diffusion probabilistic model (DDPM) in computer vision, we introduce a WCE LLIE framework based on the multi-scale convolutional neural network (CNN) and reverse diffusion process. The multi-scale design allows models to preserve high-resolution representation and context information from low-resolution, while the curved wavelet attention (CWA) block is proposed for high-frequency and local feature learning. Furthermore, we combine the reverse diffusion procedure to further optimize the shallow output and generate the most realistic image. The proposed method is compared with ten state-of-the-art (SOTA) LLIE methods and significantly outperforms quantitatively and qualitatively. The superior performance on GI disease segmentation further demonstrates the clinical potential of our proposed model.


---
## Environment

For environment setup, please follow these intructions
```
sudo apt-get install cmake build-essential libjpeg-dev libpng-dev
conda create -n llcaps python=3.9
conda activate llcaps
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

---
## Dataset
1. [Kvasir-Capsule Dataset](https://osf.io/dv2ag/)
    - [Low-light Image Pairs](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EYtX3vMBWE1KizB1scvGOkgBzG4JW5SjTMAnJuxZTUAwdg?e=KJk1k2) 
2. [Red Lesion Endoscopy Dataset](https://rdm.inesctec.pt/dataset/nis-2018-003)
    - [Low-light Image Pairs](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EZ_Dz7G4J4hBpDKn3YPng6cByGmdGt1z2Qd51fZsmv6DoA?e=aj6KlO) 
---


## Installation
For installing, follow these intructions
```
sudo apt-get install cmake build-essential libjpeg-dev libpng-dev
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
cd pytorch-gradual-warmup-lr
python setup.py install
cd ..
```


## Training

Train your model with default arguments by running

```
python train.py
```
Training arguments can be modified in 'training.yml'.

## Inference
Conduct model inference by running

```
python inference.py --input_dir /[GT_PATH] --result_dir /[GENERATED_IMAGE_PATH] --weights /[MODEL_CHECKPOINT] --save_images
```

## Evaluation (PSNR, SSIM, LPIPS)

```
python evaluation.py -dir_A /[GT_PATH] -dir_B /[GENERATED_IMAGE_PATH] 
```

---
