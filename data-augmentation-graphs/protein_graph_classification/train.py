import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, ConcatDataset, Dataset

from dataloader import ProteinDataset
from torch_geometric.loader import DataLoader

import transforms
from model import GNN_graphpred
from utils.splitters import random_split
from utils.metrics import evaluate_roc_auc, evaluate_precision
from utils.util import add_result_to_csv
import time

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(model, device, loader, optimizer, test_loader = None):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Loss matrix
        loss_mat = criterion(pred.double(), y)
        optimizer.zero_grad()
        loss = torch.mean(loss_mat)
        loss.backward()
        # print("Train loss", loss, end =" ")
        optimizer.step()

        if test_loader is not None:
            model.eval()
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                y = batch.y.view(pred.shape).to(torch.float64)

                #Loss matrix
                loss_mat = criterion(pred.double(), y)
                loss = torch.mean(loss_mat)
                # print("Test loss", loss)
                break
            model.train()

def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup(model, device, loader, optimizer, alpha):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        graph_features = model.forward_feature(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(graph_features.shape[0], -1).to(torch.float64)

        # mixup
        graph_features, y_a, y_b, lam = mixup_data(graph_features, y, alpha, device)
        pred = model.forward_predict(graph_features)
        y_a = y_a.view(pred.shape).to(torch.float64)
        y_b = y_b.view(pred.shape).to(torch.float64)

        #Loss matrix
        loss_mat = mixup_criterion(criterion, pred.double(), y_a, y_b, lam)
        optimizer.zero_grad()
        loss = torch.mean(loss_mat)
        loss.backward()

        optimizer.step()

def get_graph_sizes_and_densities(batch):
    # return a list of graph sizes in the batch
    graph_sizes = torch.zeros(len(batch.ptr)-1)
    graph_densities = torch.zeros(len(batch.ptr)-1)
    for i in range(len(batch.ptr)-1):
        num_nodes = batch.ptr[i+1] - batch.ptr[i]
        num_edges = torch.logical_and(
            batch.edge_index[1] >= batch.ptr[i], 
            batch.edge_index[1] < batch.ptr[i+1] ).sum()
        density = (2*num_edges)/(num_nodes*(num_nodes-1))
        graph_sizes[i] = num_nodes
        graph_densities[i] = density
    return graph_sizes, graph_densities

def generate_split_intervals(dataset, num_groups):
    # loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers = 4, exclude_keys=["node_labels", "node_uncertainties", "graph_labels"])

    # graph_sizes = []
    # for step, batch in enumerate(loader):
    #     graph_sizes.append(get_graph_sizes(batch))
    # graph_sizes = torch.cat(graph_sizes, dim = 0)
    graph_sizes = np.load("./notebooks/graph_sizes.npy")
    graph_densities = np.load("./notebooks/graph_densities.npy")

    # Group the graph sizes
    #   generate indexes for each graph sizes split by medians (evenly split)
    def split_by_median(idxes, arr):
        tmp_median = np.median(arr[idxes])
        group_1 = idxes[arr[idxes]<=tmp_median]
        group_2 = idxes[arr[idxes]>tmp_median]
        return group_1, group_2, tmp_median.item()

    group_idxes = [torch.arange(graph_sizes.shape[0])]
    size_intervals = []
    for _ in range(num_groups-1):
        tmp_idxes = group_idxes[0]
        group_1, group_2, tmp_median = split_by_median(tmp_idxes, graph_sizes)
        group_idxes.pop(0)
        group_idxes.append(group_1); group_idxes.append(group_2)
        size_intervals.append(tmp_median)
    size_intervals.sort()

    density_intervals = []
    for i, tmp_idxes in enumerate(group_idxes):
        tmp_density_intervals = []
        tmp_group_idxes = [tmp_idxes]
        for _ in range(num_groups-1):
            tmp_idxes = tmp_group_idxes[0]
            group_1, group_2, tmp_median = split_by_median(tmp_idxes, graph_densities)
            tmp_group_idxes.pop(0)
            tmp_group_idxes.append(group_1); tmp_group_idxes.append(group_2)
            tmp_density_intervals.append(tmp_median)
        tmp_density_intervals.sort()
        density_intervals.append(np.array(tmp_density_intervals))
    density_intervals = np.array(density_intervals)

    return size_intervals, density_intervals

def split_datasets_by_sizes(dataset, graph_size_intervals):
    '''
    Return a list of datasets bucketed by graph sizes
    '''
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers = 4, exclude_keys=["node_labels", "node_uncertainties", "graph_labels"])

    graph_sizes = []
    for _, batch in enumerate(loader):
        batch_graph_sizes, batch_graph_density = get_graph_sizes_and_densities(batch)
        graph_sizes.append(batch_graph_sizes)
    graph_sizes = torch.cat(graph_sizes, dim = 0)

    # Generate group indexes based on graph size intervals
    group_idxes = []
    group_idxes.append(graph_sizes<=graph_size_intervals[0])
    for i, tmp_size in enumerate(graph_size_intervals):
        if i < len(graph_size_intervals) - 1:
            tmp_idxes = torch.logical_and(
                graph_sizes>graph_size_intervals[i], graph_sizes<=graph_size_intervals[i+1]
            )
        else:
            tmp_idxes = graph_sizes>graph_size_intervals[i]
        group_idxes.append(tmp_idxes)
    
    for i, tmp_group_idxes in enumerate(group_idxes):
        group_idxes[i] = torch.nonzero(tmp_group_idxes, as_tuple=True)[0]
    
    # Split the dataset
    datasets = []
    for tmp_group_idxes in group_idxes:
        datasets.append(Subset(dataset, tmp_group_idxes))
    return datasets

def split_datasets_by_densities(dataset, graph_density_intervals):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers = 4, exclude_keys=["node_labels", "node_uncertainties", "graph_labels"])

    graph_densities = []
    for _, batch in enumerate(loader):
        batch_graph_sizes, batch_graph_density = get_graph_sizes_and_densities(batch)
        graph_densities.append(batch_graph_density)
    graph_densities = torch.cat(graph_densities, dim = 0)

    # Generate group indexes based on graph size intervals
    group_idxes = []
    group_idxes.append(graph_densities<=graph_density_intervals[0])
    for i, tmp_size in enumerate(graph_density_intervals):
        if i < len(graph_density_intervals) - 1:
            tmp_idxes = torch.logical_and(
                graph_densities>graph_density_intervals[i], graph_densities<=graph_density_intervals[i+1]
            )
        else:
            tmp_idxes = graph_densities>graph_density_intervals[i]
        group_idxes.append(tmp_idxes)
    
    for i, tmp_group_idxes in enumerate(group_idxes):
        group_idxes[i] = torch.nonzero(tmp_group_idxes, as_tuple=True)[0]
    
    # Split the dataset
    datasets = []
    for tmp_group_idxes in group_idxes:
        datasets.append(Subset(dataset, tmp_group_idxes))
    return datasets


def eval(model, device, loader, graph_size_intervals, graph_density_intervals=None):
    '''
    Return:
        roc_list: average AUC-ROC of each task [num_of_tasks]
        loss:     average loss [num_of_tasks]
        group_roc_auc_lists:  AUC-ROC of each group and each task [num_of_groups, num_of_tasks]
        group_losses:         average loss os each group and each task [num_of_groups, num_of_tasks]
    '''
    model.eval()
    y_true = []
    y_scores = []
    graph_sizes = []
    graph_densities = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
        batch_graph_sizes, batch_graph_density = get_graph_sizes_and_densities(batch)
        graph_sizes.append(batch_graph_sizes)
        graph_densities.append(batch_graph_density)
    
    y_true = torch.cat(y_true, dim = 0)
    y_scores = torch.cat(y_scores, dim = 0)
    graph_sizes = torch.cat(graph_sizes, dim = 0)
    graph_densities = torch.cat(graph_densities, dim = 0)
    y_true = y_true.to(torch.float64)
    
    # Loss matrix
    loss_mat = criterion(y_scores.double(), y_true)
    loss_list = loss_mat.mean(dim=0).cpu().numpy()

    roc_list = evaluate_roc_auc(preds=y_scores, target=y_true, reduction="none")
    roc_list = roc_list.cpu().numpy()

    pr_list = evaluate_precision(preds=y_scores, target=y_true, reduction="none")
    pr_list = pr_list.cpu().numpy()

    # Generate group indexes based on graph size intervals
    size_idxes_list = [graph_sizes<graph_size_intervals[0]]
    for i, tmp_size in enumerate(graph_size_intervals):
        if i < len(graph_size_intervals) - 1:
            size_idxes = torch.logical_and(
                graph_sizes>=graph_size_intervals[i], graph_sizes<graph_size_intervals[i+1]
            )
        else:
            size_idxes = graph_sizes>=graph_size_intervals[i]

        size_idxes_list.append(size_idxes)
        
    group_idxes = []
    for i, size_idxes in enumerate(size_idxes_list):
        group_idxes.append(torch.logical_and(size_idxes, graph_densities<graph_density_intervals[i][0]))
        for j, tmp_density in enumerate(graph_density_intervals[i]):
            if j < len(graph_density_intervals[i]) - 1:
                tmp_idxes = torch.logical_and(
                    size_idxes, graph_densities<graph_density_intervals[i][j+1]
                )
                tmp_idxes = torch.logical_and(
                    tmp_idxes, graph_densities>=graph_density_intervals[i][j]
                )
            else:
                tmp_idxes = torch.logical_and(size_idxes, graph_densities>=graph_density_intervals[i][j])
            group_idxes.append(tmp_idxes)
    
    # Evaluate the performance for each bucket
    group_losses = []
    group_roc_auc_lists = []
    group_pr_lists = []
    for i, tmp_group_idxes in enumerate(group_idxes):
        if tmp_group_idxes.sum() == 0: # no graph in this size interval
            group_loss = np.zeros_like(loss_list)
            group_roc_list = np.zeros_like(roc_list)
            group_pr_list = np.zeros_like(pr_list)
        else:
            group_loss = loss_mat[tmp_group_idxes].mean(dim=0).cpu().numpy()
            group_roc_list = evaluate_roc_auc(
                preds=y_scores[tmp_group_idxes], 
                target=y_true[tmp_group_idxes], 
                reduction="none").cpu().numpy()
            group_pr_list = evaluate_precision(
                preds=y_scores[tmp_group_idxes], 
                target=y_true[tmp_group_idxes], 
                reduction="none").cpu().numpy()
        group_losses.append(group_loss)
        group_roc_auc_lists.append(group_roc_list)
        group_pr_lists.append(group_pr_list)
    group_losses = np.array(group_losses)
    group_roc_auc_lists = np.array(group_roc_auc_lists)
    group_pr_lists = np.array(group_pr_lists)

    return loss_list, roc_list, pr_list, group_losses, group_roc_auc_lists, group_pr_lists

def update_metrics(task_ids, group_ids, avg_loss, avg_auroc, avg_pr, group_losses, group_aurocs, group_prs, phase="valid"):
    log = {}
    log[f"average_{phase}_loss"] = avg_loss.mean()
    log[f"average_{phase}_auroc"] = avg_auroc.mean()
    log[f"average_{phase}_precision"] = avg_pr.mean()
    for i, task_id in enumerate(task_ids):
        log[f"task_{task_id}_{phase}_loss"] = group_losses[:, i].mean()
        log[f"task_{task_id}_{phase}_auroc"] = group_aurocs[:, i].mean()
        log[f"task_{task_id}_{phase}_precision"] = group_prs[:, i].mean()
        for j, group_id in enumerate(group_ids):
            if i == 0:
                log[f"group_{group_id}_{phase}_loss"] = group_losses[j].mean()
                log[f"group_{group_id}_{phase}_auroc"] = group_aurocs[j].mean()
                log[f"group_{group_id}_{phase}_precision"] = group_prs[j].mean()
            log[f"task_{task_id}_group_{group_id}_{phase}_loss"] = group_losses[j][i]
            log[f"task_{task_id}_group_{group_id}_{phase}_auroc"] = group_aurocs[j][i]
            log[f"task_{task_id}_group_{group_id}_{phase}_precision"] = group_prs[j][i]
    return log

def save_checkpoint(model, checkpoint_dir, name = "model_best"):
    model_path = os.path.join(checkpoint_dir, f'{name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saving current model: {name}.pth ...")

def load_checkpoint(model, checkpoint_dir):
    model_path = os.path.join(checkpoint_dir, 'model_best.pth')
    model.load_state_dict(torch.load(model_path))
    print("Loading the best checkpoint!")

class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x
        
    def __len__(self):
        return len(self.subset)

def main(args):
    start = time.time()
    # setup dataset
    if args.task_idxes == -1 or args.task_idxes[0] == -1: # use all tasks
        task_idxes = list(range(10463))
    else:
        task_idxes = args.task_idxes
    
    # filtering labeling rates
    if args.labeling_rate_threshold != 0:
        labeling_rates = np.load("./notebooks/labeling_rates.npy")
        task_idxes = [task_idx for task_idx in task_idxes if labeling_rates[task_idx] > args.labeling_rate_threshold]
    num_tasks = len(task_idxes)
    print("Number of tasks: {}".format(len(task_idxes)))

    augment_function = None
    augmentation_probs = args.augmentation_probs if len(args.augmentation_probs) > 0 else None
    tree_idxes = args.tree_idxes if len(args.tree_idxes) > 0 else None
    if args.random_augment:
        augment_function = transforms.RandAugment(
            n = 2, m = args.randaug_m
        )
    elif args.use_augmentation:
        augment_function = transforms.SequentialAugmentation(
            args.augmentation_names,
            args.augmentation_ratios,
            probs=augmentation_probs,
            tree_idxes=tree_idxes
        )
    dataset = ProteinDataset(root="./dataset/Alphafold", class_idxes=task_idxes)

    if args.num_groups == 1:
        graph_size_intervals = [1e6] # put all graphs in one group
        graph_densitiy_intervals = [[0.5], [0.5]]
    else:
        graph_size_intervals, graph_densitiy_intervals = generate_split_intervals(dataset, num_groups=args.num_groups)
    print("Graph size intervals: {}".format(graph_size_intervals))
    print("Graph density intervals: {}".format(graph_densitiy_intervals))

    # random split dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed = args.seed)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(train_dataset))[:args.train_size]
    train_dataset = Subset(train_dataset, indices)

    # split the training dataset by sizes and combine specified size groups together
    if args.group_ids[0] != -1:
        group_ids = args.group_ids        
        train_datasets = split_datasets_by_sizes(train_dataset, graph_size_intervals)
        train_datasets = [split_datasets_by_densities(train_datasets[i], graph_densitiy_intervals[i]) for i in range(len(train_datasets))]
        train_datasets = sum(train_datasets, [])
        print([len(dataset) for dataset in train_datasets])
        new_datasets = []
        for group_id in args.group_ids:
            new_datasets.append(train_datasets[group_id])
        train_dataset = ConcatDataset(new_datasets)
    else:
        group_ids = list(range(args.num_groups**2))
    print("Number of training graphs: {}".format(len(train_dataset)))
    # setup dataloader
    train_dataset = TransformDataset(train_dataset, augment_function)
    valid_dataset.transform = None; test_dataset.transform = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, exclude_keys=["node_labels", "node_uncertainties", "graph_labels"])
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, exclude_keys=["node_labels", "node_uncertainties", "graph_labels"])
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, exclude_keys=["node_labels", "node_uncertainties", "graph_labels"])

    # set up model
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = GNN_graphpred(args.num_layer, args.emb_dim, args.num_node_type, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    model.to(device)

    log_metrics = {}
    for run in range(1, args.runs+1):
        # reset parameters of model
        model.reset_parameters()

        # set up optimizer: different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.weight_decay)
        
        # create checkpoint dir
        # task_idxes_str = "_".join([str(idx) for idx in task_idxes])
        # if len(task_idxes_str) > 100:
        #     task_idxes_str = task_idxes_str[:100]
        augment_str = "_".join(
                    [f"{name}_{ratio}" for (name, ratio) in zip(args.augmentation_names, args.augmentation_ratios)]
                )
        group_str = "_".join([str(idx) for idx in group_ids])
        checkpoint_dir = "./saved_results/{}_group_{}_{}".format(args.gnn_type, group_str, augment_str) + \
            (f"_mixup_{args.mixup_alpha}" if args.train_mixup else "") + (f"_randaug_{args.randaug_m}" if args.random_augment else "")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        # start training
        best_val_auroc = 0
        for epoch in range(1, args.epochs+1):
            if args.train_mixup:
                train_mixup(model, device, train_loader, optimizer, args.mixup_alpha)
            else:
                train(model, device, train_loader, optimizer)

            train_loss, train_auroc, train_pr, train_group_losses, train_group_aurocs, train_group_prs = \
                eval(model, device, train_loader, graph_size_intervals, graph_densitiy_intervals)
            val_loss, val_auroc, val_pr, val_group_losses, val_group_aurocs, val_group_prs = \
                eval(model, device, valid_loader, graph_size_intervals, graph_densitiy_intervals)
            test_loss, test_auroc, test_pr, test_group_losses, test_group_aurocs, test_group_prs = \
                eval(model, device, test_loader, graph_size_intervals, graph_densitiy_intervals)
            print("====epoch %d train loss: %f val loss: %f test loss: %f" %(epoch, train_loss.mean(), val_loss.mean(), test_loss.mean()))
            print("====         train auroc: %f val auroc: %f test auroc: %f"    %(train_auroc.mean(), val_auroc.mean(), test_auroc.mean()))
            print("====         train precision: %f val precision: %f test precision: %f"    %(train_pr.mean(), val_pr.mean(), test_pr.mean()))

            if val_group_aurocs[group_ids].mean() > best_val_auroc:
                best_val_auroc = val_group_aurocs[group_ids].mean()
                save_checkpoint(model, checkpoint_dir)

        # evaluate metrics with the best model checkpoint
        if args.epochs > 0 or args.load_model_dir == '':
            load_checkpoint(model, checkpoint_dir)
        else:
            load_model_dir = f"./saved_results/{args.load_model_dir}"
            load_checkpoint(model, load_model_dir)
        # Use graph size intervals as one element list to compute metrics on all graphs
        train_loss, train_auroc, train_pr, train_group_losses, train_group_aurocs, train_group_prs = \
            eval(model, device, train_loader, graph_size_intervals, graph_densitiy_intervals)
        val_loss, val_auroc, val_pr, val_group_losses, val_group_aurocs, val_group_prs = \
            eval(model, device, valid_loader, graph_size_intervals, graph_densitiy_intervals)
        test_loss, test_auroc, test_pr, test_group_losses, test_group_aurocs, test_group_prs = \
            eval(model, device, test_loader, graph_size_intervals, graph_densitiy_intervals)
        
        val_log = update_metrics(task_idxes, list(range(args.num_groups**2)), val_loss, val_auroc, val_pr, val_group_losses, val_group_aurocs, val_group_prs, phase="valid")
        test_log = update_metrics(task_idxes, list(range(args.num_groups**2)), test_loss, test_auroc, test_pr, test_group_losses, test_group_aurocs, test_group_prs, phase="test")
        
        for key, val in val_log.items():
            if key in log_metrics:
                log_metrics[key].append(val)
            else:
                log_metrics[key] = [val, ]

        for key, val in test_log.items():
            if key in log_metrics:
                log_metrics[key].append(val)
            else:
                log_metrics[key] = [val, ]

        # record the model metrics
        print(
            f"Run: {run:2.0f} train auroc: {train_auroc.mean():.4f} valid auroc: {val_auroc.mean():.4f} test auroc: {test_auroc.mean():.4f}",
            f"valid group aurocs: {val_group_aurocs.mean(axis=1)} test group aurocs: {test_group_aurocs.mean(axis=1)}",
            f"Run: {run:2.0f} train precision: {train_pr.mean():.4f} valid precision: {val_pr.mean():.4f} test precision: {test_pr.mean():.4f}",
            f"valid group precisions: {val_group_prs.mean(axis=1)} test group precisions: {test_group_prs.mean(axis=1)}"
        )

    for key, val in log_metrics.items():
        log_metrics[key] = np.array(val)
    
    print("Valid AUC-ROC: {:.4f}+/-{:.4f} Test AUC-ROC: {:.4f}+/-{:.4f}".format(
        np.mean(log_metrics["average_valid_auroc"]), np.std(log_metrics["average_valid_auroc"]),
        np.mean(log_metrics["average_test_auroc"]), np.std(log_metrics["average_test_auroc"]),
    ))

    print("Valid Precision: {:.4f}+/-{:.4f} Test Precision: {:.4f}+/-{:.4f}".format(
        np.mean(log_metrics["average_valid_precision"]), np.std(log_metrics["average_valid_precision"]),
        np.mean(log_metrics["average_test_precision"]), np.std(log_metrics["average_test_precision"]),
    ))

    end = time.time()
    print("Total time: {:.4f}s".format(end-start))

    # save results to csv files
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    # for task_id in task_idxes:
    for group_id in group_ids:
        # save results for each group of each task
        result_datapoint = {
            # "Task": task_id, 
            "Group": group_id,
            # "Trained on (tasks)": task_idxes,
            "Trained on (groups)": group_ids,
            "Augmentation": augment_str,
            "Probs": "_".join([str(prob) for prob in augmentation_probs]) if augmentation_probs is not None else "None",
            "Tree": "_".join([str(idx) for idx in tree_idxes]) if tree_idxes is not None else "None",                
        }
        # Add average task 
        for phase in ["valid", "test"]: 
            result_datapoint[f"average_{phase}_loss"] = np.mean(log_metrics[f"average_{phase}_loss"])
            result_datapoint[f"average_{phase}_auroc"] = np.mean(log_metrics[f"average_{phase}_auroc"])
            result_datapoint[f"average_{phase}_precision"] = np.mean(log_metrics[f"average_{phase}_precision"])
            # result_datapoint[f"task_average_{phase}_loss"] = np.mean(log_metrics[f"task_{task_id}_{phase}_loss"])
            # result_datapoint[f"task_average_{phase}_auroc"] = np.mean(log_metrics[f"task_{task_id}_{phase}_auroc"])
            # result_datapoint[f"task_average_{phase}_precision"] = np.mean(log_metrics[f"task_{task_id}_{phase}_precision"])
            result_datapoint[f"group_average_{phase}_loss"] = np.mean(log_metrics[f"group_{group_id}_{phase}_loss"])
            result_datapoint[f"group_average_{phase}_auroc"] = np.mean(log_metrics[f"group_{group_id}_{phase}_auroc"])
            result_datapoint[f"group_average_{phase}_precision"] = np.mean(log_metrics[f"group_{group_id}_{phase}_precision"])

            result_datapoint[f"average_{phase}_loss_std"] = np.std(log_metrics[f"average_{phase}_loss"])
            result_datapoint[f"average_{phase}_auroc_std"] = np.std(log_metrics[f"average_{phase}_auroc"])
            result_datapoint[f"average_{phase}_precision_std"] = np.std(log_metrics[f"average_{phase}_precision"])
            result_datapoint[f"group_average_{phase}_loss_std"] = np.std(log_metrics[f"group_{group_id}_{phase}_loss"])
            result_datapoint[f"group_average_{phase}_auroc_std"] = np.std(log_metrics[f"group_{group_id}_{phase}_auroc"])
            result_datapoint[f"group_average_{phase}_precision_std"] = np.std(log_metrics[f"group_{group_id}_{phase}_precision"])
        # for key, vals in log_metrics.items():
        #     if f"task_{task_id}_group_{group_id}" in key:
        #         metric_name = "_".join(key.split("_")[-2:])
        #         result_datapoint[metric_name] = np.mean(vals)
        #         result_datapoint[metric_name+"_std"] = np.std(vals)
        file_name = os.path.join(file_dir, "{}.csv".format(args.save_name))
        add_result_to_csv(result_datapoint, file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    ''' Training '''
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--train_size', type=int, default=30000)
    
    ''' Model '''
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--num_node_type', type=int, default=20, 
                        help="number of types of nodes")
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = '', 
                        help='filename to read the model (if there is any)')
    parser.add_argument('--num_workers', type=int, default = 4, 
                        help='number of workers for dataset loading')
    parser.add_argument('--load_model_dir', type=str, default = '', 
                        help='directory to load the model')

    ''' Dataset '''
    parser.add_argument('--task_idxes', type=int, nargs='+', default=-1)
    parser.add_argument('--labeling_rate_threshold', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42, 
                        help = "Seed for splitting the dataset")
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument("--save_name", type=str, default="test", 
                        help="The name for the training results csv file")

    ''' Size generalization '''
    parser.add_argument('--num_groups', type=int, default=4)
    parser.add_argument('--group_ids', type=int, nargs='+', default=[-1])

    ''' Augmentation '''
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--augmentation_names', nargs = "+", type=str, default=["Identity"], help='augmentation names')
    parser.add_argument('--augmentation_ratios', nargs = "+", type=float, default=[0.0], help='augmentation ratios')
    parser.add_argument('--augmentation_probs', nargs = "+", type=float, default=[], help='augmentation probs')
    parser.add_argument('--tree_idxes', nargs='+', type=int, default=[], help='tree index')

    parser.add_argument('--train_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=1)

    parser.add_argument('--random_augment', action='store_true')
    parser.add_argument('--randaug_m', type=float, default=0.1)

    args = parser.parse_args()
    main(args)