import argparse
import collections
import os
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
# from trainer.iwebcam_trainer import IWebCamTrainer
from trainer import *
from utils import prepare_device, add_result_to_csv, deep_copy
from utils.linear_evaluation import LinearEvaluation
from torchvision import transforms
from transforms import TestTransform
from transforms.compositions import SimCLRTransfrom, SimCLR, CompositeTransform
from transforms.rand_augment import RandAugment
from model.modeling_vit import VisionTransformer, CONFIGS

from third_party.data_augmentation.lisa import LISACutMix, LISAMixUp
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.grouper import CombinatorialGrouper
from torchvision.transforms import Compose, ToTensor
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time

def get_group_subset(dataset, split, group_id, frac=1.0, transform=None):
    """
    Args:
        - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                        Must be in self.split_dict.
        - frac (float): What fraction of the split to randomly sample.
                        Used for fast development on a small dataset.
        - transform (function): Any data transformations to be applied to the input x.
    Output:
        - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
    """
    if split not in dataset.split_dict:
        raise ValueError(f"Split {split} not found in dataset's split_dict.")

    split_mask = dataset.split_array == dataset.split_dict[split]
    # group_mask = dataset.metadata_array[:, 0] == group_id
    group_mask = np.array([(group in group_id) for group in dataset.metadata_array[:, 0]])
    split_mask = np.logical_and(split_mask, group_mask)
    split_idx = np.where(split_mask)[0]

    if frac < 1.0:
        # Randomly sample a fraction of the split
        num_to_retain = int(np.round(float(len(split_idx)) * frac))
        split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

    return WILDSSubset(dataset, split_idx, transform)

def main(config, args):
    start = time.time()
    logger = config.get_logger('train')
    
    if args.dataset == "iwildcam":
        pre_transforms = [transforms.Resize((448, 448))]
        post_transforms = [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    elif args.dataset == "camelyon17":
        pre_transforms = [transforms.Resize((96, 96))]
        post_transforms = [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    if args.dataset == "iwildcam":
        dataset = get_dataset(dataset=args.dataset, download=True)
    else:
        raise ValueError(f"dataset : {args.dataset} not implemented.")
    
    if args.train_bilevel:
        # load best augment combination
        best_augments = {name : [] for name in args.group_id}
        best_ratios = {name : [] for name in args.group_id}
        best_probs = {name : [] for name in args.group_id}
        best_idxes = {name : [] for name in args.group_id}
        for id in args.group_id:
            file_dir = os.path.join("./specified_augmentations/", f'iwildcam_{id}.txt')
            with open(file_dir, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        augmentations = line.strip().split(" ")
                        best_augments[id] = augmentations
                    elif i == 1:
                        ratios = line.strip().split(" ")
                        best_ratios[id] = [float(ratio) for ratio in ratios]
                    elif i == 2:
                        probs = line.strip().split(" ")
                        best_probs[id] = [float(prob) for prob in probs]
                    elif i == 3:
                        idxes = line.strip().split(" ")
                        best_idxes[id] = [int(idx) for idx in idxes]
                print(augmentations, ratios, probs, idxes)

        train_datasets = []; task_to_train_loaders = {}
        for group_id in args.group_id:
            if args.train_no_transforms:
                transform = CompositeTransform(transform_names=[], ratios=[], probs=None, tree_idxes=None,
                                        pre_transforms=pre_transforms, post_transforms=post_transforms)
            else:
                transform = CompositeTransform(transform_names=best_augments[group_id], ratios=best_ratios[group_id], 
                                        probs=best_probs[group_id], tree_idxes=best_idxes[group_id], 
                                        pre_transforms=pre_transforms, post_transforms=post_transforms)
                
            test_transform = transforms.Compose(pre_transforms + [ToTensor()] + post_transforms)
            
            train_dataset = get_group_subset(dataset, 'train', [int(group_id)], transform=transform)
            train_data_loader = module_data.BaseDataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2)

            train_datasets.append(train_data_loader.dataset)
            task_to_train_loaders[group_id] = train_data_loader
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        valid_dataset = get_group_subset(dataset, 'id_val', args.group_id, transform=test_transform)
        test_dataset = get_group_subset(dataset, 'id_test', args.group_id, transform=test_transform)

        train_data_loader = module_data.BaseDataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2)
        valid_data_loader = module_data.BaseDataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=2)
        test_data_loader = module_data.BaseDataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=2)
    else:
        # set basic transform
        augmentation_probs = args.augmentation_probs if len(args.augmentation_probs) > 0 else None
        tree_idxes = args.tree_idxes if len(args.tree_idxes) > 0 else None
        transform = CompositeTransform(transform_names=args.augmentation_names, ratios=args.augmentation_ratios, 
                                    probs=augmentation_probs, tree_idxes=tree_idxes, 
                                    pre_transforms=pre_transforms, post_transforms=post_transforms)
        test_transform = transforms.Compose(pre_transforms + [ToTensor()] + post_transforms)
        
        if args.dataset == "iwildcam":
            train_dataset = get_group_subset(dataset, 'train', args.group_id, transform=transform)
            valid_dataset = get_group_subset(dataset, 'id_val', args.group_id, transform=test_transform)
            test_dataset = get_group_subset(dataset, 'id_test', args.group_id, transform=test_transform)

            train_data_loader = module_data.BaseDataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2)
            valid_data_loader = module_data.BaseDataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=2)
            test_data_loader = module_data.BaseDataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=2)

    assert 0 < args.data_frac <= 1
    if args.data_frac < 1:
        train_data_len = len(train_data_loader.sampler)
        train_data_loader.sampler.indices = train_data_loader.sampler.indices[:int(train_data_len*args.data_frac)]
    logger.info("Train Size: {} Valid Size: {} Test Size: {}".format(
        len(train_data_loader.sampler), 
        len(valid_data_loader.sampler), 
        len(test_data_loader.sampler)))
    
    # build model architecture, then print to console
    if args.is_vit:
        vit_config = CONFIGS[args.vit_type]
        model = config.init_obj('arch', module_arch, config = vit_config, img_size = args.img_size, zero_head=True)
        model.load_from(np.load(args.vit_pretrained_dir))
    else:
        model = config.init_obj('arch', module_arch)
    logger.info(model)

    test_metrics = {}

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    source_state_dict = deep_copy(model.state_dict())
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

     # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    augment_str = "_".join(
                [f"{name}_{ratio}" for (name, ratio) in zip(args.augmentation_names, args.augmentation_ratios)]
            )
    
    load_model_dir = os.path.join("./saved/models", args.load_model_dir) \
        if args.load_model_dir is not None else None
    
    for run in range(args.runs):
        model.reset_parameters(source_state_dict)
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        checkpoint_dir = "./saved/models/{}_{}_{}_{}/".format(
            args.dataset, config["arch"]["type"], args.group_id, augment_str) + ("_bilevel" if args.train_bilevel else "")\
            + ("_dro" if args.train_bilevel else "")
        
        if load_model_dir and os.path.exists(os.path.join(load_model_dir, "model_best.pth")):
            model.load_state_dict(
                torch.load(os.path.join(load_model_dir, "model_best.pth"))["state_dict"]
            )

        if args.train_bilevel:
            trainer = BilevelSupervisedTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir,
                            train_data_loaders=task_to_train_loaders,
                            weight_lr=args.weight_lr, collect_gradient_step=args.collect_gradient_step, update_weight_step = args.update_weight_step
                            )
        else:
            trainer = Trainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir
                            )
        # model.load_state_dict(torch.load("./saved/models/simclr_resnet50_100_simclr/model_epoch_100.pth")["state_dict"])
        trainer.train()
        test_log = trainer.test(use_val=True)
        test_log.update(trainer.test())

        for key, val in test_log.items():
            if key in test_metrics:
                test_metrics[key].append(val)
            else:
                test_metrics[key] = [val, ]

    # print training results
    for key, vals in test_metrics.items():
        logger.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(vals), np.std(vals)))

    # save results into .csv
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # save test results
    result_datapoint = {
        "Augmentation": augment_str,
        "Probs": "_".join([str(prob) for prob in augmentation_probs]) if augmentation_probs is not None else "None",
        "Tree": "_".join([str(idx) for idx in tree_idxes]) if tree_idxes is not None else "None",
    }
    for key, vals in test_metrics.items():
        result_datapoint[key] = np.mean(vals)
        result_datapoint[key+"_std"] = np.std(vals)
    file_name = os.path.join(file_dir, "{}_test.csv".format(args.save_name))
    add_result_to_csv(result_datapoint, file_name)

    end = time.time()
    logger.info(f"Total Time: {end - start}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, nargs='+',
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--dataset', type=str, default="iwildcam", help='dataset name')
    args.add_argument('--group_id', nargs = "+", type=int, default=[0], help='group id')
    args.add_argument('--batch_size', type=int, default=16, help='batch size')

    args.add_argument('--temperature', default=0.5, type=float)
    args.add_argument('--world_size', default=1, type=int)
    args.add_argument('--data_frac', type=float, default=1.0)

    args.add_argument('--train_simclr', action='store_true')
    args.add_argument('--train_randaugment', action='store_true')
    args.add_argument('--randaugment_n', type=int, default=2)
    args.add_argument('--randaugment_m', type=int, default=10)

    args.add_argument('--augmentation_names', nargs = "+", type=str, default=["Identity"], help='augmentation names')
    args.add_argument('--augmentation_ratios', nargs = "+", type=float, default=[0.2], help='augmentation ratios')
    args.add_argument('--augmentation_probs', nargs = "+", type=float, default=[], help='augmentation probs')
    args.add_argument('--tree_idxes', nargs = "+", type=int, default=[], help='tree idxes')

    args.add_argument('--is_vit', action='store_true')
    args.add_argument('--img_size', type=int, default=224)
    args.add_argument("--vit_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args.add_argument("--vit_pretrained_dir", type=str, default="pretrained/imagenet21k_ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    
    args.add_argument('--save_name', type=str, default="test", help='save name')
    args.add_argument('--runs', type=int, default=2, help='number of runs')

    args.add_argument('--train_no_transforms', action="store_true")
    args.add_argument('--train_bilevel', action="store_true")
    args.add_argument('--weight_lr', type=float, default=1)
    args.add_argument('--collect_gradient_step', type=float, default=1)
    args.add_argument('--update_weight_step', type=float, default=50)

    args.add_argument('--load_model_dir', type=str, default=None, help='load model dir')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        # CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;args;encoder_name"),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--n_gpu'], type=int, target='n_gpu'),
        CustomArgs(['--n_classes'], type=int, target='arch;args;n_classes'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
