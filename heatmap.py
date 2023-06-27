import torch
import cv2
from PIL import Image
from skimage import exposure, img_as_float, io 
import os
import random
import numpy as np

img_gt = cv2.imread('YOUR IMAGE PATH')	
img_gt = torch.tensor(img_gt)
old_path = r"YOUR_PATH"
new_path = r"YOUR_PATH"
file_list = os.walk(old_path)
for root, dirs, files in file_list:
    for file in files:
        pic_path = os.path.join(root, file)
        img_lq = cv2.imread(pic_path)
        img_lq = torch.tensor(img_lq)
        diff_lq = torch.mul(torch.abs(torch.sub(img_gt, img_lq)), 10)
        diff_lq = diff_lq.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_lq = (diff_lq * 255.0).round().astype(np.uint8)
        diff_lq_color = cv2.applyColorMap(diff_lq, cv2.COLORMAP_JET)
        cv2.imwrite(new_path+'/'+file, diff_lq_color)
        print('processed:',new_path+'/'+file)