import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from networks.LLCaps import LLCaps
from dataloaders.data_rgb import get_validation_data
import utils
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='Image Enhancement using MIRNet')

parser.add_argument('--input_dir', default='Kvasir_Capsule_LLIE/eval', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results/Kvasir_Capsule_LLIE', type=str, help='Directory for results')
parser.add_argument('--weights', default='checkpoints/Kvasir_Capsule_LLIE/best.pth', type=str, help='Path to weights')
model_restoration = LLCaps()
parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save Enahnced images in the result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()

with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        filenames = data_test[2]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored,0,1)
     
        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                enhanced_img = img_as_ubyte(rgb_restored[batch])
                utils.save_img(args.result_dir +'/'+ filenames[batch][:-4] + '.png', enhanced_img)
            
psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
print("PSNR: %.2f " %(psnr_val_rgb))

