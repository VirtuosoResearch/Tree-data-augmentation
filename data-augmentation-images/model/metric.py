import torch
import numpy as np

def macro_f1(output, target):
    from torcheval.metrics.functional import multiclass_f1_score
    with torch.no_grad():
        macro_f1 = multiclass_f1_score(output, target ,num_classes=182, average="macro")
        return macro_f1.cpu()

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
