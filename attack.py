#!/usr/bin/env python
# coding: utf-8

# configuration:
# models_ver - insert YOLO version's numbers that the UAP will be trained on.
# 
# epsilon, lambda_1, lambda_2 - attack's parameters. more information can be found in the [paper](https://arxiv.org/abs/2205.13618)
# 
# BDD_IMG_DIR - a path to the BDD validation set images (or any other wanted dataset)
# 
# BDD_LAB_DIR - a path to the BDD validation set labels (or any other wanted dataset)

# In[1]:


models_vers = [3,4,5,6] # for example: models_vers = [5] or models_vers = [3, 4, 5]
epsilon = 32
lambda_1 = 1
lambda_2 = 10
seed = 42
patch_size=(640,640)
img_size=(640,640)
batch_size = 1
num_workers = 1
max_labels_per_img = 65
BDD_IMG_DIR = 'BDD_train'
BDD_LAB_DIR = 'BDD_train_label'


# Load BDD dataset:

# In[16]:


import torch
import os
import random
import numpy

from uap_attack.augmentations1 import train_transform
from uap_attack.split_data_set_combined import SplitDatasetCombined_BDD

def collate_fn(batch):
    return tuple(zip(*batch))

def set_random_seed(seed_value, use_cuda=True):
    numpy.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


split_dataset = SplitDatasetCombined_BDD(
            img_dir= BDD_IMG_DIR,
            lab_dir= BDD_LAB_DIR,
            max_lab=max_labels_per_img,
            img_size=img_size,
            transform=train_transform,
            collate_fn=collate_fn)

train_loader, val_loader, test_loader = split_dataset(val_split=0.1,
                                                      shuffle_dataset=True,
                                                      random_seed=seed,
                                                      batch_size=batch_size,
                                                      ordered=False,
                                                      collate_fn=collate_fn)


# create UAP:

# In[ ]:


import numpy
from uap_attack.gen_uap import GEN_UAP

torch.cuda.empty_cache()

torch.autograd.set_detect_anomaly(True)

patch_name = r"yolov"
for ver in models_vers:
  patch_name += f"_{ver}"
patch_name += f"_epsilon={epsilon}_lambda1={lambda_1}_lambda2={lambda_2}"

uap_attack = GEN_UAP(patch_folder=patch_name, train_loader=train_loader, val_loader=val_loader, epsilon = epsilon, lambda_1=lambda_1, lambda_2=lambda_2, patch_size=patch_size, models_vers=models_vers)
adv_img = uap_attack.run_attack()


