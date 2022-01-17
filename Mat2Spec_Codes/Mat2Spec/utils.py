import numpy as np
import torch
import pandas as pd
import os
import shutil
import argparse
from operator import attrgetter

from   sklearn.model_selection import train_test_split
from   sklearn.metrics import mean_absolute_error as sk_MAE
from   tabulate import tabulate
import random,time

def set_device(gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    return device

def set_model_properties(crystal_property):
    if crystal_property   in ['poisson-ratio','band-gap','absolute-energy','fermi-energy','formation-energy']:
        norm_action   = None; classification = None
    elif crystal_property == 'is_metal':
        norm_action   = 'classification-1'; classification = 1
    elif crystal_property == 'is_not_metal':
        norm_action   = 'classification-0'; classification = 1
    else:    
        norm_action   = 'log'; classification = None
    return norm_action, classification

def torch_MAE(tensor1,tensor2):
    return torch.mean(torch.abs(tensor1-tensor2))

def torch_accuracy(pred_tensor,true_tensor):
    _,pred_tensor   = torch.max(pred_tensor,dim=1)
    correct         = (pred_tensor==true_tensor).sum().float()
    total           = pred_tensor.size(0)
    accuracy_ans    = correct/total
    return accuracy_ans

def output_training(metrics_obj,epoch,estop_val,extra='---'):
    header_1, header_2 = 'MSE | e-stop','MAE | TIME'
    if metrics_obj.c_property in ['is_metal','is_not_metal']:
        header_1,header_2     = 'Cross_E | e-stop','Accuracy | TIME'

    train_1,train_2 = metrics_obj.training_loss1[epoch],metrics_obj.training_loss2[epoch]
    valid_1,valid_2 = metrics_obj.valid_loss1[epoch],metrics_obj.valid_loss2[epoch]
    
    tab_val = [['TRAINING',f'{train_1:.4f}',f'{train_2:.4f}'],['VALIDATION',f'{valid_1:.4f}',f'{valid_2:.4f}'],['E-STOPPING',f'{estop_val}',f'{extra}']]
    
    output = tabulate(tab_val,headers= [f'EPOCH # {epoch}',header_1,header_2],tablefmt='fancy_grid')
    print(output)
    return output

def load_metrics():
    saved_metrics = pickle.load(open("MODELS/metrics_.pickle", "rb", -1))
    return saved_metrics


def freeze_params(model, params_to_freeze_list):
    for str in params_to_freeze_list:
        attr = attrgetter(str)(model)
        attr.requires_grad = False
        attr.grad = None


def unfreeze_params(model, params_to_unfreeze_list):
    for str in params_to_unfreeze_list:
        attr = attrgetter(str)(model)
        #print(str)
        #print(attr)
        attr.requires_grad = True


def RobustL1(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    absolute = torch.abs(output - target)
    loss = np.sqrt(2.0) * absolute * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    squared = torch.pow(output - target, 2.0)
    loss = 0.5 * squared * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)
