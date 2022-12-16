#Imports
import csv
import gzip
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras import backend as K
from torch.optim import lr_scheduler
from torchvision import models
from sklearn import metrics



class CategoryDataset(Dataset):
    """Dataset for polyvore with 5 categories(upper, bottom, shoe, bag, accessory),
    each suit has at least 3 items, the missing items will use a mean image.
    Args:
        root_dir: Directory stores source images
        data_file: A json file stores each outfit index and description
        data_dir: Directory stores data_file and mean_images
        transform: Operations to transform original images to fix size
        use_mean_img: Whether to use mean images to fill the blank part
        neg_samples: Whether generate negative sampled outfits
    """
    def __init__(self,
                 root_dir="/content/drive/MyDrive/ML_Final_Project/data/images/",
                 data_file="/content/drive/MyDrive/ML_Final_Project/data/train_no_dup_with_category_3more_name.json",
                 data_dir="/content/drive/MyDrive/ML_Final_Project/data/",
                 transform=None,
                 use_mean_img=True,
                 neg_samples=True):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.use_mean_img = use_mean_img
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]
        self.neg_samples = neg_samples # if True, will randomly generate negative outfit samples
    
        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.word_to_idx)
        self.vocabulary.append('UNK')
        with open(os.path.join(self.data_dir, 'final_word_dict.txt')) as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.word_to_idx)
                    self.vocabulary.append(name)


    def __getitem__(self, index):
        """It could return a positive suits or negative suits"""
        set_id, parts = self.data[index]
        if random.randint(0, 1) and self.neg_samples:
            #to_change = random.sample(list(parts.keys()), 3) # random choose 3 negative items
            to_change = list(parts.keys()) # random choose negative items
        else:
            to_change = []
        imgs = []
        labels = []
        names = []
        for part in ['upper', 'bottom', 'shoe', 'bag', 'accessory']:
            if part in to_change: # random choose a image from dataset with same category
                choice = self.data[index]
                while (choice[0] == set_id) or (part not in choice[1].keys()):
                    choice = random.choice(self.data)
                img_path = os.path.join(self.root_dir, str(choice[0]), str(choice[1][part]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(choice[1][part]['name'])))
                labels.append('{}_{}'.format(choice[0], choice[1][part]['index']))
            elif part in parts.keys():
                img_path = os.path.join(self.root_dir, str(set_id), str(parts[part]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(parts[part]['name'])))
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
            elif self.use_mean_img:
                img_path = os.path.join(self.data_dir, part+'.png')
                names.append(torch.LongTensor([])) # mean_img embedding
                labels.append('{}_{}'.format(part, 'mean'))
            else:
                continue
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        input_images = torch.stack(imgs)
        is_compat = (len(to_change)==0)

        offsets = list(itertools.accumulate([0] + [len(n) for n in names[:-1]]))
        offsets = torch.LongTensor(offsets)
        return input_images, names, offsets, set_id, labels, is_compat

    def __len__(self):
        return len(self.data)

    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK']
            for w in name.split()]

def collate_fn(data):
    """Need custom a collate_fn"""
    data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images,  names, offsets, set_ids, labels, is_compat = zip(*data)
    lengths = [i.shape[0] for i in images]
    is_compat = torch.LongTensor(is_compat)
    names = sum(names, [])
    offsets = list(offsets)
    images = torch.stack(images)
    return (
        lengths,
        images,
        names,
        offsets,
        set_ids,
        labels,
        is_compat
    )
