import utils
import models_mae
import mlp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time


## DEFINE THE MODEL

class CLS_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
              
        self.input_fc = nn.Linear(input_dim, 500)
        self.hidden_fc = nn.Linear(500, 100)
        self.output_fc = nn.Linear(100, output_dim)
        
    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1) # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x)) # h_1 = [batch size, 250]
        h_2 = F.relu(self.hidden_fc(h_1)) # h_2 = [batch size, 100]
        y_pred = self.output_fc(h_2) # y_pred = [batch size, output dim]

        return y_pred


## DEFINE UTILS
def my_forward_encoder(model, img, mask_ratio):
    # x = torch.tensor(img)
    x = img
    
    x = x.float()
    
    # embed patches
    x = model.patch_embed(x)

    # add pos embed w/o cls token
    x = x + model.pos_embed[:, 1:, :]

    # masking: length -> length * mask_ratio
    x, mask, ids_restore = model.random_masking(x, mask_ratio=mask_ratio)
    
    # append cls token
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    
    # apply Transformer blocks
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    
    return x, mask, ids_restore 

def latent_before_decoder(model, x, ids_restore, device):
 
    x_cls = x[:, 0, :]

    return x_cls

def get_decoders_input(model, img, mask_ratio, device):

  x, mask, ids_restore = my_forward_encoder(model, img, mask_ratio)
  num_unmasked = x.shape[1]-1
  y = latent_before_decoder(model, x, ids_restore, device)

  return y, ids_restore, num_unmasked




## TRAIN THE MODEL

def calculate_accuracy(y_pred, y):
    _, top_pred = torch.max(y_pred, 1)
    correct = (top_pred==y).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, mae_model, iterator, optimizer, criterion, mask_ratio, device):

    epoch_loss = 0
    epoch_acc = 0
    mlp_time_vec = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        
        x = x.to(device)
        y = y.to(device)
                
        optimizer.zero_grad()

        x, ids_restore, num_unmasked = get_decoders_input(mae_model, x, mask_ratio, device)

        y_pred = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, mae_model, iterator, criterion, mask_ratio, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            
            x = x.to(device)
            y = y.to(device)

            x, ids_restore, num_unmasked = get_decoders_input(mae_model, x, mask_ratio, device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



## EXAMINE THE MODEL
def get_predictions(model, mae_model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            x, ids_restore, num_unmasked = get_decoders_input(mae_model, x, 0, device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs

def get_representations(model, iterator, device):

    model.eval()

    outputs = []
    intermediates = []
    labels = []

    with torch.no_grad():

        for (x, y) in tqdm(iterator):

            x = x.to(device)

            y_pred, h = model(x)

            outputs.append(y_pred.cpu())
            intermediates.append(h.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    intermediates = torch.cat(intermediates, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, intermediates, labels