#!/usr/bin/env python
# coding: utf-8

"""
Test code written by Viresh Ranjan

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""

import copy
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import MincountLoss, PerturbationLoss
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import torch.optim as optim
import json
import csv
import torch.nn.functional as nnf



parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'annotation_Val_Test_384_VarV2.json'
data_split_file = data_path + 'Train_Test_Val_FSC147_HW6_Split.json'
im_dir = data_path + 'images_384_VarV2'

mask_file_dir = data_path + 'mask_images'

if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
if use_gpu: resnet50_conv.cuda()
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean')
regressor.load_state_dict(torch.load(args.model_path))
if use_gpu: regressor.cuda()
regressor.eval()

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)


cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors
#all_ids = []
#all_outputs = []
#data = {}

print("Evaluation on {} data".format(args.test_split))
im_ids = data_split[args.test_split]
pbar = tqdm(im_ids)

for im_id in pbar:
    anno = annotations[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    rects = list()
    for bbox in bboxes:
        x1, y1 = bbox[0][0], bbox[0][1]
        x2, y2 = bbox[2][0], bbox[2][1]
        rects.append([y1, x1, y2, x2])

    image = Image.open('{}/{}'.format(im_dir, im_id))
    image.load()
    sample = {'image': image, 'lines_boxes': rects}

    sample = Transform(sample) #changes dimensions of image
    image, boxes = sample['image'], sample['boxes']

    if use_gpu:
        image = image.cuda()
        boxes = boxes.cuda()

    #load in appropriate mask
    name_temp = im_id.split(".")
    name = name_temp[0] + "_anno.png"
    mask_img = Image.open('{}/{}'.format(mask_file_dir, name))
    mask_img.load()
    mask_img = np.array(mask_img)

    #change values to 0 or 1
    mask_img = mask_img/255.0 #white = [255,255,255] and black = [0,0,0] --> black is important region
    #get mask image in same form as input
    mask_img = np.moveaxis(mask_img, -1, 0)

    with torch.no_grad(): features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

    if not args.adapt:
        #print("\nin this one!!!!!!!!!!!!!!!!")
        with torch.no_grad(): output = regressor(features)
    else:
        #print("\n OTHER ONE")
        features.required_grad = True
        adapted_regressor = copy.deepcopy(regressor)
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)
        for step in range(0, args.gradient_steps):
            optimizer.zero_grad()
            output = adapted_regressor(features)

            #alter mask shape to fit output shape
            new_shape = torch.Size([output.shape[2], output.shape[3]])
            mask_img_tensor = torch.from_numpy(mask_img[0])

            mask_img_tensor = mask_img_tensor.unsqueeze(0)
            mask_img_tensor = mask_img_tensor.unsqueeze(0)

            temp_mask = nnf.interpolate(mask_img_tensor, size=new_shape, mode = 'bicubic')

            mask_img_tensor = mask_img_tensor.squeeze(0)
            mask_img_tensor = mask_img_tensor.squeeze(0)
            if use_gpu:
              temp_mask = temp_mask.cuda()
            #when mask is 1, sum over density values
            #sum_error = 0
            c = temp_mask*output
            sum_error = torch.sum(c)
            #for i in range(0,len(temp_mask[0][0])):
            #  for j in range(0,len(temp_mask[0][0][0])):
            #    if(temp_mask[0][0][i][j] == 1):
                
            #      if(output[0][0][i][j]>0.0):
                    #print("IN HERE")
            #        sum_error = sum_error + output[0][0][i][j]
            #args.weight_mincount = args.weight_mincount * 1.05
            #args.weight_perturbation = args.weight_perturbation * 1.05
            lCount = args.weight_mincount * MincountLoss(output, boxes)
            lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
            #mask error is of same magnitude
            lmask_error = (1e-09)*sum_error
            Loss = lCount + lPerturbation + lmask_error


            # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
            # So Perform gradient descent only for non zero cases
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()
        features.required_grad = False
        output = adapted_regressor(features)

    #print("THIS IS OUTPUT")
    #print(im_id)
    #print(output)
 
    gt_cnt = dots.shape[0]
    pred_cnt = output.sum().item()
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2

    #all_ids.append(im_id)
    #all_outputs.append(output.tolist())
    #print(output.tolist())
    #data[im_id] = output.tolist()

    pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.\
                         format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
    print("")

#numpy.set_printoptions(threshold=sys.maxsize)
#print(all_outputs)

#with open('eggs.csv','w', newline='') as csvfile:
#  spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
#  for i in range(0,len(all_ids)):
#    spamwriter.writerow([im_id[i]] + [all_outputs[i]])
  #spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
  #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
#np.savetxt("part2_density_output.txt", data)
#create txt file with this data
#with open("part2_density_output.txt", "w") as the_file:
#  json.dump(data, the_file)
#  for i in range(0,len(all_ids)):
  #  the_file.write(str(all_ids[i]) +" " + str(list(all_outputs[i])) + "\n")
  #the_file.close()

print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
