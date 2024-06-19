import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from utils import setup_logging_logic, get_logger, k_fold, add_result_to_csv
from utils.linear_evaluation import LinearEvaluation
from datasets import get_tu_dataset
from torch_geometric.loader import DataLoader
from models import ResGCN, GNN_graphpred
from trainer import *

import transforms
import models.losses as module_loss
import models.metrics as module_metric
import time
from torch.utils.data import Subset, ConcatDataset


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

def main(args):
    setup_logging_logic()
    logger = get_logger("main")
    start = time.time()   
    # set up dataloaders: 1. get augmentations 2. get dataset 3. k-fold split 4. create dataloaders
    augmentation_probs = args.augmentation_probs if len(args.augmentation_probs) > 0 else None
    tree_idxes = args.tree_idxes if len(args.tree_idxes) > 0 else None
    if args.train_simclr:
        augment_func = transforms.SimCLRTransfrom(
            args.augmentation_names, 
            args.augmentation_ratios,
            augmentation_probs,
            tree_idxes
        )
    elif args.train_randaugment:
        augment_func = transforms.RandAugment(args.rand_augment_n, args.rand_augment_m)
    else:
        augment_func = getattr(transforms, args.augmentation_name)(args.augmentation_ratio)

    # Replace degree feats for REDDIT datasets (less redundancy, faster).
    if  args.dataset in [
            'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
        args.feat_str = args.feat_str.replace('odeg100', 'odeg10')
    # Replace degree and akx feats for dd (less redundancy, faster).
    if args.dataset in ['DD']:
        args.feat_str = args.feat_str.replace('odeg100', 'odeg10')
        args.feat_str = args.feat_str.replace('ak3', 'ak1')
    dataset = get_tu_dataset(args.dataset, args.feat_str, args.root, augment_func)

    if args.num_groups == 1:
        graph_size_intervals = [1e6] # put all graphs in one group
        graph_densitiy_intervals = [[0.5], [0.5]]
    else:
        graph_size_intervals, graph_densitiy_intervals = generate_split_intervals(dataset, num_groups=args.num_groups)

    # split the training dataset by sizes and combine specified size groups together
    if args.num_groups != 1 and args.group_ids[0] != -1:
        group_ids = args.group_ids        
        train_datasets = split_datasets_by_sizes(dataset, graph_size_intervals)
        train_datasets = [split_datasets_by_densities(train_datasets[i], graph_densitiy_intervals[i]) for i in range(len(train_datasets))]
        train_datasets = sum(train_datasets, [])
        print([len(dataset) for dataset in train_datasets])
        new_datasets = []
        for group_id in args.group_ids:
            new_datasets.append(train_datasets[group_id])
        train_dataset = ConcatDataset(new_datasets)
    else:
        group_ids = list(range(args.num_groups**2))

    train_indices, test_indices, val_indices = k_fold(dataset, args.folds, args.epoch_select, args.semi_split)
    train_loaders = []; val_loaders = []; test_loaders = []
    for fold_idx in range(args.folds):
        train_dataset = dataset.copy(train_indices[fold_idx])
        test_dataset = dataset.copy(test_indices[fold_idx])
        val_dataset = dataset.copy(val_indices[fold_idx])

        print("Train size, val size, test size: {}, {}, {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        train_loaders.append(train_loader); val_loaders.append(val_loader); test_loaders.append(test_loader)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # get function handles of loss and metrics
    criterion = getattr(module_loss, args.loss_name)
    metrics = [getattr(module_metric, met) for met in args.metrics]

    log_metrics = {}
    for fold_idx, (train_data_loader, valid_data_loader, test_data_loader) in \
        enumerate(zip(train_loaders, val_loaders, test_loaders)):

        if fold_idx in args.fold_idxes:
            # build model architecture, then print to console; create a model each time
            if args.model == "ResGCN":
                use_xg = "xg" in dataset[0]
                model = ResGCN(in_channels=dataset.num_features,
                            hidden_channels=args.hidden_channels, 
                            out_channels=dataset.num_classes,
                            num_feat_layers=args.n_layers_feat, 
                            num_conv_layers=args.n_layers_conv,
                            num_fc_layers=args.n_layers_fc, 
                            residual=args.skip_connection, 
                            gfn=False, collapse=False,
                            res_branch=args.res_branch, global_pool=args.global_pool, 
                            dropout=args.dropout, edge_norm=args.edge_norm, 
                            use_xg=use_xg, 
                            xg_size=dataset[0].xg.size(1) if use_xg else 0)
            elif args.model == "GIN":
                model = GNN_graphpred(
                        input_dim = dataset.num_features,
                        num_layer = args.n_layers, 
                        emb_dim = args.hidden_channels, 
                        num_tasks = dataset.num_classes,
                        drop_ratio = args.dropout, 
                        graph_pooling=args.global_pool,
                        JK=args.JK,
                        residual=args.skip_connection,
                        gnn_type = "gin"   
                    )
            elif args.model == "GCN":
                model = GNN_graphpred(
                        input_dim = dataset.num_features,
                        num_layer = args.n_layers, 
                        emb_dim = args.hidden_channels, 
                        num_tasks = dataset.num_classes,
                        drop_ratio = args.dropout, 
                        graph_pooling=args.global_pool,
                        JK=args.JK,
                        residual=args.skip_connection,
                        gnn_type = "gcn"   
                    )
            model_path = os.path.join(f"./saved/models/{args.dataset}/{args.feat_str}", args.model_path)
            print(model_path, os.path.exists(model_path))
            if os.path.exists(model_path) and args.model_path != "":
                model.load_state_dict(torch.load(model_path))
            model = model.to(device)

            # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma) 

            augment_str = "_".join(
                    [f"{name}_{ratio}" for (name, ratio) in zip(args.augmentation_names, args.augmentation_ratios)]
                )
            if args.train_simclr:
                checkpoint_dir = \
                    f"./saved/models/{args.dataset}/{args.feat_str}/contra/{augment_str}" 
                trainer = SimCLRTrainer(model, optimizer, lr_scheduler, criterion, metrics, 
                                train_loader=train_data_loader,
                                valid_loader=valid_data_loader,
                                test_loader=test_data_loader,
                                device=device,
                                logger=logger,
                                epochs=args.epochs,
                                save_epochs = args.save_epochs,
                                mnt_metric=args.mnt_metric,
                                mnt_mode=args.mnt_mode,
                                checkpoint_dir=checkpoint_dir
                                )
            elif args.train_randaugment:
                checkpoint_dir = \
                    f"./saved/models/{args.dataset}/{args.feat_str}/contra/rand_augment"
                trainer = SimCLRTrainer(model, optimizer, lr_scheduler, criterion, metrics, 
                                train_loader=train_data_loader,
                                valid_loader=valid_data_loader,
                                test_loader=test_data_loader,
                                device=device,
                                logger=logger,
                                epochs=args.epochs,
                                save_epochs = args.save_epochs,
                                mnt_metric=args.mnt_metric,
                                mnt_mode=args.mnt_mode,
                                checkpoint_dir=checkpoint_dir
                                )
            else:
                checkpoint_dir = f"./saved/models/{args.dataset}/{args.feat_str}/finetune/{args.augmentation_name}"
                trainer = Trainer(model, optimizer, lr_scheduler, criterion, metrics, 
                                train_loader=train_data_loader,
                                valid_loader=valid_data_loader,
                                test_loader=test_data_loader,
                                device=device,
                                logger=logger,
                                epochs=args.epochs,
                                save_epochs = args.save_epochs,
                                mnt_metric=args.mnt_metric,
                                mnt_mode=args.mnt_mode,
                                checkpoint_dir=checkpoint_dir
                                )

            trainer.train()
            test_log = trainer.test(phase="test")
            val_log = trainer.test(phase="valid")
            test_log.update(val_log)

        # linear evaluation on every fold
        evaluator = LinearEvaluation(model, 
                    train_dataset=train_data_loader.dataset, 
                    valid_dataset=valid_data_loader.dataset,
                    test_dataset=test_data_loader.dataset,
                    device=device,
                    state_dict_dir=checkpoint_dir,
                    state_dict_name="model_best")
        eval_log = evaluator.eval()
        test_log.update(eval_log)
        
        for key, val in test_log.items():
            if key in log_metrics:
                log_metrics[key].append(val)
            else:
                log_metrics[key] = [val, ]

    # print training results
    for key, vals in log_metrics.items():
        logger.info("{}: {:.2f} +/- {:.2f}".format(key, np.mean(vals)*100, np.std(vals)*100))
    
    # save results into .csv
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    # save test results
    result_datapoint = {
            "Task": args.dataset,
            "Augmentation": augment_str,
            "Probs": "_".join([str(prob) for prob in augmentation_probs]) if augmentation_probs is not None else "None",
            "Tree": "_".join([str(idx) for idx in tree_idxes]) if tree_idxes is not None else "None",
        }
    for key, vals in log_metrics.items():
        result_datapoint[key] = np.mean(vals)
        result_datapoint[key+"_std"] = np.std(vals)
    file_name = os.path.join(file_dir, "{}.csv".format(args.save_name))
    add_result_to_csv(result_datapoint, file_name)
    end = time.time()
    print("Running time: {}".format(end - start))

if __name__ == "__main__":
    ''' Write args in argparse'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="NCI1", help='dataset name')
    parser.add_argument('--feat_str', type=str, default="deg+odeg100", help='feature string')
    parser.add_argument('--root', type=str, default="./data", help='dataset root')
    parser.add_argument('--folds', type=int, default=10, help='number of folds')
    parser.add_argument('--epoch_select', type=str, default="valid_max", help='epoch selection method')
    parser.add_argument('--semi_split', type=int, default=10, help='number of semi-supervised samples')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--fold_idxes', type=int, default=[0,1,2,3,4,5,6,7,8,9], nargs='+', help='fold index')

    # model params
    parser.add_argument('--model', type=str, default="ResGCN", help='model name')

    #    GIN params
    parser.add_argument('--n_layers', type=int, default=5, help='number of layers')
    parser.add_argument('--JK', type=str, default="last", help='JK type')

    #    ResGCN params
    parser.add_argument('--hidden_channels', type=int, default=128, help='hidden channels')
    parser.add_argument('--n_layers_feat', type=int, default=1, help='number of feature layers')
    parser.add_argument('--n_layers_conv', type=int, default=3, help='number of convolution layers')
    parser.add_argument('--n_layers_fc', type=int, default=2, help='number of fully connected layers')
    parser.add_argument('--skip_connection', action="store_true", help='skip connection')
    parser.add_argument('--res_branch', type=str, default="BNConvReLU", help='residual branch')
    parser.add_argument('--global_pool', type=str, default="sum", help='global pooling')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--edge_norm', type=bool, default=True, help='edge normalization')

    parser.add_argument('--loss_name', type=str, default="nll_loss", help='loss name')
    parser.add_argument('--metrics', nargs = "+", type=str, default=["accuracy"], help='metrics')
    parser.add_argument('--model_path', type=str, default="", help='model path')
    parser.add_argument('--device', type=int, default=0, help='device')

    # augment params
    parser.add_argument('--augmentation_name', type=str, default="Identity", help='augmentation name')
    parser.add_argument('--augmentation_ratio', type=float, default=0.0, help='augmentation ratio')

    parser.add_argument('--train_simclr', action="store_true", help='train simclr')
    parser.add_argument('--augmentation_names', nargs = "+", type=str, default=["DropNodes"], help='augmentation names')
    parser.add_argument('--augmentation_ratios', nargs = "+", type=float, default=[0.2], help='augmentation ratios')
    parser.add_argument('--augmentation_probs', nargs = "+", type=float, default=[], help='augmentation probs')
    parser.add_argument('--tree_idxes', nargs='+', type=int, default=[], help='tree index')

    parser.add_argument('--train_randaugment', action="store_true", help='train randaugment')
    parser.add_argument('--rand_augment_n', type=int, default=2,)
    parser.add_argument('--rand_augment_m', type=float, default=0.2)

    # training params
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--save_epochs', type=int, default=10, help='save epochs')
    parser.add_argument('--mnt_metric', type=str, default="val_accuracy", help='monitor metric')
    parser.add_argument('--mnt_mode', type=str, default="max", help='monitor mode')
    parser.add_argument('--lr_decay_step', type=int, default=50, help='learning rate decay step')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='learning rate decay gamma')

    ''' Size generalization '''
    parser.add_argument('--num_groups', type=int, default=0)
    parser.add_argument('--group_ids', type=int, nargs='+', default=[-1])

    # save name
    parser.add_argument('--save_name', type=str, default="test", help='save name')
    args = parser.parse_args()

    main(args)