

<div align="center">

<samp>

<h2> LLCaps: Learning to Illuminate Low-Light Capsule Endoscopy with Curved Wavelet Attention and Reverse Diffusion </h1>

<h4> Long Bai*, Tong Chen*, Yanan Wu, An Wang, Mobarakol Islam, and Hongliang Ren </h3>

<h3> Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023 (Oral) </h2>

</samp>   

| **[[```arXiv```](<https://arxiv.org/abs/2307.02452>)]** | **[[```Paper```](<https://link.springer.com/chapter/10.1007/978-3-031-43999-5_4>)]** |
|:-------------------:|:-------------------:|

---

</div>     

If you find our code, paper, or dataset useful, please cite the paper as

```bibtex
@inproceedings{bai2023llcaps,
  title={LLCaps: Learning to Illuminate Low-Light Capsule Endoscopy with Curved Wavelet Attention and Reverse Diffusion},
  author={Bai, Long and Chen, Tong and Wu, Yanan and Wang, An and Islam, Mobarakol and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={34--44},
  year={2023},
  organization={Springer}
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
    - [Low-light Image Pairs](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EYtX3vMBWE1KizB1scvGOkgBzG4JW5SjTMAnJuxZTUAwdg?e=gbdyuR)
    - [External Validation Set](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EcmsZ2NJKSNDk-jKmpSfp1sB_2h1v-2ZlI9Bfu8v4Y4hIA?e=pyNFHS)
2. [Red Lesion Endoscopy Dataset](https://rdm.inesctec.pt/dataset/nis-2018-003)
    - [Low-light Image Pairs](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EZ_Dz7G4J4hBpDKn3YPng6cByGmdGt1z2Qd51fZsmv6DoA?e=veMC5d)
    - [RLE Segmentation Set](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EeAhx_FEHLJMv1zXNb8oS_YBbN-Une6U7g2v2KOx2BYPcA?e=ENUuTk) (You may match the segmentation masks with the images by the filenames.)
---

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
