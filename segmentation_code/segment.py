from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time
from medpy import metric
# from evaluation_func import *
#from ploting import VisdomLinePlotter
#from visdom import Visdom

def ConfusionMatrix(numClass, imgPredict, Label):  
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return precision  

def Recall(confusionMatrix):
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score

def IntersectionOverUnion(confusionMatrix):  
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def resize255to1(predict_img):
    h, w = predict_img.shape
    for i in range(h):
        for j in range(w):
            if predict_img[i][j] >= 1:
                predict_img[i][j] = 1
    return predict_img



#######################################################
#Checking if GPU is used
#######################################################

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################

batch_size = 4
print('batch_size = ' + str(batch_size))

valid_size = 0.15

epoch = 100 #"epoch"
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

#plotter = VisdomLinePlotter(env_name='Tutorial Plots')

#######################################################
#Setting up the model
#######################################################

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]


def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

#passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary


model_test = model_unet(model_Inputs[0], 3, 1)

model_test.to(device)

#######################################################
#Getting the Summary of Model
#######################################################

torchsummary.summary(model_test, input_size=(3, 128, 128))

#######################################################
#Passing the Dataset of Images and Labels
#######################################################

#data
t_data = 'train_img_path'
test_folderP = 'test_img_path'
#test
test_image = 'one_image_sample'
test_label = 'one_mask_sample'
#mask
l_data = 'train_mask_path'
test_folderL = 'test_mask_path'

Training_Data = Images_Dataset_folder(t_data,
                                      l_data)



data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)


initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) 

MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)


folder_name = 'result'

New_folder = './' + folder_name

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)


read_pred = './' + folder_name + '/pred'


if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)

try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)


read_model_path = './'+ folder_name +'/Unet_D_' + str(epoch) + '_' + str(batch_size)

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()
    scheduler.step(i)
    lr = scheduler.get_lr()

    model_test.train()
    k = 1

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        input_images(x, y, i, n_iter, k)

       
        opt.zero_grad()

        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)     # Dice_loss Used

        train_loss += lossT.item() * x.size(0)
        lossT.backward()
     
        opt.step()
        x_size = lossT.item() * x.size(0)
        k = 2


    model_test.eval()
    torch.no_grad() 

    for x1, y1 in valid_loader:
        x1, y1 = x1.to(device), y1.to(device)

        y_pred1 = model_test(x1)
        lossL = calc_loss(y_pred1, y1)     # Dice_loss Used

        valid_loss += lossL.item() * x1.size(0)
        x_size1 = lossL.item() * x1.size(0)

    im_tb = Image.open(test_image)
    im_label = Image.open(test_label).convert('RGB')
    s_tb = data_transform(im_tb)
    s_label = data_transform(im_label)
    s_label = s_label.detach().numpy()

    pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()

    x1 = plt.imsave(
        './'+ folder_name +'/pred/img_iteration_' + str(n_iter) + '_epoch_'
        + str(i) + '.png', pred_tb[0][0])

 
    train_loss = train_loss / len(train_idx)
    valid_loss = valid_loss / len(valid_idx)

    if (i+1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))

    if valid_loss <= valid_loss_min and epoch_valid >= i: 

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(),'./'+ folder_name +'/Unet_D_' +
                                              str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                                              + '_batchsize_' + str(batch_size) + '.pth')
       
        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid+1
        valid_loss_min = valid_loss



    x1 = torch.nn.ModuleList(model_test.children())
    x2 = len(x1)
    dr = LayerActivations(x1[x2-1])
    img = Image.open(test_image)
    s_tb = data_transform(img)

    pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()

    plot_kernels(dr.features, n_iter, 7, cmap="rainbow")

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    n_iter += 1

#######################################################
#closing the tensorboard writer
#######################################################



test1 =model_test.load_state_dict(torch.load('./'+folder_name+'/Unet_D_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))




if torch.cuda.is_available():
    torch.cuda.empty_cache()



model_test.load_state_dict(torch.load('./'+folder_name+'/Unet_D_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))

model_test.eval()

read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)  # To sort


read_test_folder112 = './'+ folder_name +'/gen_images'


if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)

try:
    os.mkdir(read_test_folder112)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)


#For Prediction Threshold

read_test_folder_P_Thres = './'+ folder_name +'/pred_threshold'


if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold

read_test_folder_L_Thres = './' + folder_name + '/label_threshold'


if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)


img_test_no = 0

for i in range(len(read_test_folder)):
    im = Image.open(x_sort_test[i])

    im1 = im
    im_n = np.array(im1)
    im_n_flat = im_n.reshape(-1, 1)

    for j in range(im_n_flat.shape[0]):
        if im_n_flat[j] != 0:
            im_n_flat[j] = 255

    s = data_transform(im)
    pred = model_test(s.unsqueeze(0).cuda()).cpu()
    pred = F.sigmoid(pred)
    pred = pred.detach().numpy()


    if i % 24 == 0:
        img_test_no = img_test_no + 1

    x1 = plt.imsave('./' + folder_name + '/gen_images/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', pred[0][0])


data_transform = torchvision.transforms.Compose([
             torchvision.transforms.Grayscale(),])



read_test_folderP = glob.glob('./' + folder_name + '/gen_images/*')
x_sort_testP = natsort.natsorted(read_test_folderP)


read_test_folderL = glob.glob(test_folderL) # label
x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort


dice_score123 = 0.0
x_count = 0
x_dice = 0
total_acc = 0
total_prec = 0
total_sen = 0
total_f1_score = 0
total_miou = 0
total_fwiou = 0
total_dice = 0
total_jc = 0
total_hd = 0
total_asd = 0


for i in range(len(read_test_folderP)):
    print(len(read_test_folderP))
    x = Image.open(x_sort_testP[i])
    s = data_transform(x)
    s = np.array(s)
    s = threshold_predictions_v(s)

    #save the images
    x1 = plt.imsave('./' + folder_name + '/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s)

    y = Image.open(x_sort_testL[i])
    s2 = data_transform(y)
    s3 = np.array(s2) 

    y1 = plt.imsave('./' + folder_name + '/label_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s3)

    total = dice_coeff(s, s3)
    s_01 = resize255to1(s)
    s_label_01 = resize255to1(s3)


    confusion_matrix = ConfusionMatrix(2, s_01, s_label_01)
 
    acc = OverallAccuracy(confusion_matrix)
    prec = np.mean(Precision(confusion_matrix))
    sen = np.mean(Recall(confusion_matrix))
    miou = MeanIntersectionOverUnion(confusion_matrix)
    fw_iou = Frequency_Weighted_Intersection_over_Union(confusion_matrix)
    dice = metric.binary.dc(s_01, s_label_01)
    jc = metric.binary.jc(s_01, s_label_01)
    hd = metric.binary.hd95(s_01, s_label_01)
    asd = metric.binary.asd(s_01, s_label_01)


    print(x_sort_testP[i])
    total_miou = total_miou + miou
    total_dice = total_dice + dice
    total_hd = total_hd + hd


print('mIoU: ' + str(total_miou/len(read_test_folderP)))
print('Dice: ' + str(total_dice/len(read_test_folderP)))
print('HD: ' + str(total_hd/len(read_test_folderP)))



