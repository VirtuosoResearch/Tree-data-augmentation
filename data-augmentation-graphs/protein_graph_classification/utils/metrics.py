import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torchmetrics.functional.classification import f1_score, accuracy, auroc, average_precision

''' Evaluation metrics '''
def evaluate_roc_auc(preds, target, reduction="mean"):
    if len(target.shape) == 1:
        target = target.unsqueeze(-1)

    rocauc_list = auroc(preds, target.type(torch.long), num_classes=target.shape[1], average=None)
    if reduction == "mean":
        return rocauc_list.mean()
    else:
        if len(rocauc_list.shape) == 0:
            rocauc_list = rocauc_list.unsqueeze(-1)
        return rocauc_list

def evaluate_f1(preds, target, reduction="mean"):
    if len(target.shape) == 1:
        target = target.unsqueeze(-1)

    f1_list = f1_score(preds, target.type(torch.long), num_classes=target.shape[1], average=None)
    if reduction == "mean":
        return f1_list.mean()
    else:
        return f1_list

def evaluate_accuracy(preds, target, reduction="mean"):
    if len(target.shape) == 1:
        target = target.unsqueeze(-1)
    
    acc_list = accuracy(preds, target.type(torch.long), num_classes=target.shape[1], average=None)
    if reduction == "mean":
        return acc_list.mean()
    else:
        return acc_list

def evaluate_precision(preds, target, reduction="mean"):
    if len(target.shape) == 1:
        acc_list = average_precision(preds, target.type(torch.long), num_classes=1, average=None)
        acc_list = [acc_list]
    else:
        acc_list = average_precision(preds, target.type(torch.long), num_classes=target.shape[1], average=None)
        if target.shape[1] == 1:
            acc_list = [acc_list]
    acc_list = torch.Tensor(acc_list)
    if reduction == "mean":
        return acc_list.mean()
    else:
        return acc_list

def evaluate_mse(preds, target, reduction="mean"):
    mses = F.mse_loss(preds, target, reduction='none')
    if reduction == "mean":
        return mses.mean()
    else:
        return mses
