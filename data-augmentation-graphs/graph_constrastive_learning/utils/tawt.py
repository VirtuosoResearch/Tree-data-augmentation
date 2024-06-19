import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def get_average_feature_gradients(model, train_loader, criterion, device, step = 1):
    loss = 0
    count = 0
    for i, (data_1, data_2) in enumerate(train_loader):
        if i >= step:
            break
        # modify model forward function
        data_1, data_2 = data_1.to(device), data_2.to(device)

        out1 = model.forward_cl(data_1) 
        out2 = model.forward_cl(data_2)
        loss = criterion(out1, out2)
        loss += loss
        count += 1
    loss = loss/count
    if hasattr(model, "module"):
        feature_gradients = grad(loss, model.module.parameters(), retain_graph=False, create_graph=False,
                                allow_unused=True)
    else:
        # modify taking the gradients of the encoder
        feature_gradients = grad(loss, model.parameters(), retain_graph=False, create_graph=False,
                                allow_unused=True)
    feature_gradients = torch.cat([gradient.view(-1) for gradient in feature_gradients if gradient is not None]) # flatten gradients
    return feature_gradients

def get_task_weights_gradients_multi(model, source_loaders, criterion, device, step=1):
    source_gradients = {}
    for task, task_train_loader in source_loaders.items():
        task_gradients = get_average_feature_gradients(model, task_train_loader, criterion, device, step)
        source_gradients[task] = task_gradients
    
    # average source gradients to a target gradient:
    target_gradients = torch.zeros_like(task_gradients)
    count = 0
    for task, task_gradient in source_gradients.items():
        target_gradients = target_gradients + task_gradient
        count += 1
    target_gradients = target_gradients/count

    num_tasks = len(source_loaders.keys())
    task_weights_gradients = torch.zeros((num_tasks, ), device=device, dtype=torch.float)
    for i, task in enumerate(source_loaders.keys()):
        task_weights_gradients[i] = -F.cosine_similarity(target_gradients, source_gradients[task], dim=0)
    return task_weights_gradients