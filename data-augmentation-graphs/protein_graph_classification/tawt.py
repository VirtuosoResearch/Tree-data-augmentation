import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def get_average_feature_gradients(model, train_loader, criterion, device, step = 1, return_vector=True):
    loss = 0
    count = 0
    for i, batch in enumerate(train_loader):
        if i >= step:
            break
        # modify model forward function
        batch = batch.to(device)
        pred = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        loss_mat = criterion(pred.double(), y)
        loss = torch.mean(loss_mat)
        loss += loss
        count += 1
    loss = loss/count
    # modify taking the gradients of the encoder
    params = [ p for p in model.parameters() if p.requires_grad ]
    feature_gradients = grad(loss, params, retain_graph=False, create_graph=False,
                            allow_unused=True)
    if return_vector:
        feature_gradients = torch.cat([gradient.view(-1) for gradient in feature_gradients if gradient is not None]) # flatten gradients
        return feature_gradients
    else:
        return [gradient for gradient in feature_gradients if gradient is not None]


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    # if len(w) != len(v):
    #     raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
    first_grads = [grad for grad in first_grads if grad is not None]
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=False, allow_unused=True)

    return [grad for grad in return_grads if grad is not None]


def s_test(v, model, source_loaders, criterion, device, damp=0.01, scale=25.0,
           recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    h_estimate = v

    task_loader_list = list(source_loaders.values())
    task_num = len(task_loader_list)

    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        train_loader = task_loader_list[i % task_num]
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(pred.shape).to(torch.float64)
            loss_mat = criterion(pred.double(), y)
            loss = torch.mean(loss_mat)

            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)

            # Recursively caclulate h_estimate
            h_estimate = [
                 _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        # display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate

def get_task_weights_gradients_multi_hessian(model, source_loaders, criterion, device,
                damp=0.01, scale=100, recursion_depth=32, r=1):
    '''
    Calculate s on a training dataset
    '''
    source_gradients = {}
    for task, task_train_loader in source_loaders.items():
        task_gradients = get_average_feature_gradients(model, task_train_loader, criterion, device, step=1, return_vector=False)
        source_gradients[task] = task_gradients
    
    # average source gradients to a target gradient:
    target_gradients = [torch.zeros_like(gradient) for gradient in task_gradients if gradient is not None]
    count = 0
    for task, task_gradient in source_gradients.items():
        for i, gradient in enumerate(task_gradient):
            target_gradients[i] = target_gradients[i] + gradient
        count += 1
    target_gradients = [gradient/count for gradient in target_gradients]

    # compute s
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(target_gradients, model, source_loaders, criterion,
                                      device = device, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))
        # display_progress("Averaging r-times: ", i, r)

    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        for j in range(len(s_test_vec)):
            s_test_vec[j] += s_test_vec_list[i][j]

    s_test_vec = [i / r for i in s_test_vec]

    # compute s times group gradients
    for task, task_gradient in source_gradients.items():
        source_gradients[task] =  torch.cat([gradient.view(-1) for gradient in task_gradient if gradient is not None]) 
    s = torch.cat([gradient.view(-1) for gradient in s_test_vec if gradient is not None])

    num_tasks = len(source_loaders.keys())
    task_weights_gradients = torch.zeros((num_tasks, ), device=device, dtype=torch.float)
    for i, task in enumerate(source_loaders.keys()):
        task_weights_gradients[i] = -F.cosine_similarity(s, source_gradients[task], dim=0)
    return task_weights_gradients

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