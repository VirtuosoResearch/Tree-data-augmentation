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
from trainer import *
from utils import prepare_device, add_result_to_csv, deep_copy
from utils.linear_evaluation import LinearEvaluation
from torchvision import transforms
from transforms.compositions import SimCLRTransfrom, SimCLR
from transforms.simclr_augment import SimCLRTestTransfrom
from transforms.rand_augment import RandAugment
from transforms.lisa import LISAMixUp
import copy
import pandas
import pandas as pd
import time

from data_loader.multitask_dataset import MultitaskDataset

def main(config, args):
    start = time.time()
    logger = config.get_logger('train')

    # load best augment combination
    # read .csv, get highest "test_accuracy" ("Messidor", "Aptos", "Jinchi") 
    best_augments = {"Messidor" : [], "Aptos" : [], "Jinchi" : []}
    best_ratios = {"Messidor" : [], "Aptos" : [], "Jinchi" : []}
    best_probs = {"Messidor" : [], "Aptos" : [], "Jinchi" : []}
    best_idxes = {"Messidor" : [], "Aptos" : [], "Jinchi" : []}
    for dataset_name in args.datasets:
        file_dir = os.path.join("./specified_augmentations/", f'{dataset_name}.txt')
        with open(file_dir, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    augmentations = line.strip().split(" ")
                    best_augments[dataset_name] = augmentations
                elif i == 1:
                    ratios = line.strip().split(" ")
                    best_ratios[dataset_name] = [float(ratio) for ratio in ratios]
                elif i == 2:
                    probs = line.strip().split(" ")
                    best_probs[dataset_name] = [float(prob) for prob in probs]
                elif i == 3:
                    idxes = line.strip().split(" ")
                    best_idxes[dataset_name] = [int(idx) for idx in idxes]
            print(augmentations, ratios, probs, idxes)

    # setup data_loader instances
    task_to_transforms = {}
    task_to_train_datasets = {}
    task_to_train_loaders = {}
    task_to_valid_loaders = {}
    task_to_test_loaders = {}

    # loading multi datasets
    for dataset_name in args.datasets:
        config['data_loader']['type'] = dataset_name + "DataLoader"
        if config['data_loader']['type'] == "MessidorDataLoader":
            pre_transforms = [
                    transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                    transforms.Pad((0, 37, 0, 38)),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                ]
            post_transforms = [transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613])]
        elif config['data_loader']['type'] == "JinchiDataLoader":
            pre_transforms = [
                    transforms.Resize((224, 224)), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                ]
            post_transforms = [transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613])]
        elif config['data_loader']['type'] == "AptosDataLoader":
            pre_transforms = [
                    transforms.Resize(224), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                ]
            post_transforms = [transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613])]
        else:
            print("Unknown data loader type!")

        if args.train_no_transforms:
            transform = SimCLRTransfrom(transform_names=[], ratios=[], 
                                        probs=None, tree_idxes=None, 
                                        pre_transforms=pre_transforms, post_transforms=post_transforms)
        elif args.train_simclr:
            transform = SimCLR(pre_transforms=pre_transforms, post_transforms=post_transforms)
        elif args.train_randaugment:
            pre_transforms.append(RandAugment(args.randaugment_n, args.randaugment_m))
            pre_transforms += [
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(),
                ]
            transform = SimCLRTransfrom(pre_transforms=pre_transforms, post_transforms=post_transforms,
                                         probs=None, tree_idxes=None)
        else: # train the selected augments from tree
            pre_transforms += [
                    transforms.RandomResizedCrop((224, 224)),
                    transforms.RandomHorizontalFlip(),
                ]
            
            transform = SimCLRTransfrom(transform_names=best_augments[dataset_name], ratios=best_ratios[dataset_name], 
                                        probs=best_probs[dataset_name], tree_idxes=best_idxes[dataset_name], 
                                        pre_transforms=pre_transforms, post_transforms=post_transforms)
        
        # add transform to this dataset.
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.2, test_split=0.2, phase = "train", transform=transform)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()

        task_to_transforms[dataset_name] = transform
        task_to_train_datasets[dataset_name] = train_data_loader.dataset
        task_to_train_loaders[dataset_name] = train_data_loader
        task_to_valid_loaders[dataset_name] = valid_data_loader
        task_to_test_loaders[dataset_name] = test_data_loader
        logger.info("Train dataset size: {} Valid dataset size: {} Test dataset size: {}".format(
                    len(train_data_loader.sampler), 
                    len(valid_data_loader.sampler),
                    len(test_data_loader.sampler)))

    # load multitask joined dataloader
    multitask_train_dataloader = module_data.load_multitask_dataloader(task_to_train_datasets, config['data_loader']['args']['batch_size'])

    model = config.init_obj('arch', module_arch, vit_type=args.vit_type, img_size=args.img_size, vit_pretrained_dir=args.vit_pretrained_dir,
                             tasks=args.datasets)
    logger.info(model)

    test_metrics = {}

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    source_state_dict = deep_copy(model.state_dict())

    for run in range(args.runs):
        # reset dataset transforms:
        for dataset_name in args.datasets:
            task_to_train_loaders[dataset_name].dataset.transform = task_to_transforms[dataset_name]
            task_to_valid_loaders[dataset_name] = task_to_train_loaders[dataset_name].split_validation()
            task_to_test_loaders[dataset_name] = task_to_train_loaders[dataset_name].split_test()
        
        model.reset_parameters(source_state_dict)
        # get function handles of loss and metrics
        criterion = module_loss.NT_Xent(config["data_loader"]["args"]["batch_size"], args.temperature, args.world_size)
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        checkpoint_dir = "./saved/models/simclr_{}_{}_{}_multi/".format(
        config["arch"]["args"]["encoder_name"], 
        config["trainer"]["epochs"],
        "simclr" if args.train_simclr else ("randaugment" if args.train_randaugment else "custom")) 

        if args.train_bilevel:
            trainer = BilevelTrainer(model, criterion, metrics, optimizer,
                                config=config,
                                device=device,
                                train_data_loader=train_data_loader,
                                valid_data_loader=valid_data_loader,
                                test_data_loader=test_data_loader,
                                lr_scheduler=lr_scheduler,
                                checkpoint_dir=checkpoint_dir,
                                multitask_train_dataloader = multitask_train_dataloader,
                                train_data_loaders = task_to_train_loaders,
                                # valid_data_loaders = task_to_valid_loaders,
                                # test_data_loaders = task_to_test_loaders,   
                                weight_lr=args.weight_lr, collect_gradient_step=args.collect_gradient_step,
                                update_weight_step = args.update_weight_step
                            ) # [1, 0.5, 0.1, 0.05, 0.01]
        else:
            trainer = MultitaskSimCLRTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir,
                            multitask_train_dataloader = multitask_train_dataloader,
                            train_data_loaders = task_to_train_loaders,
                            valid_data_loaders = task_to_valid_loaders,
                            test_data_loaders = task_to_test_loaders,            
                            )

        trainer.train()
        test_log = trainer.test()
        
        for task_name in task_to_train_loaders.keys():
            if task_name == "Messidor":
                supervised_train_transform = transforms.Compose([
                            transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                            transforms.Pad((0, 37, 0, 38)),
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                        ])
                supervised_test_transform = transforms.Compose([
                            transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                            transforms.Pad((0, 37, 0, 38)),
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                        ])
            elif task_name == "Jinchi":
                supervised_train_transform = transforms.Compose([
                            transforms.Resize((224, 224)), 
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                        ])
                supervised_test_transform = transforms.Compose([
                            transforms.Resize((224, 224)), 
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                        ])
            elif task_name == "Aptos":
                supervised_train_transform = transforms.Compose([
                            transforms.Resize(224), 
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                        ])
                supervised_test_transform = transforms.Compose([
                            transforms.Resize(224), 
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                        ])
            
            task_to_train_loaders[task_name].dataset.transform = supervised_train_transform
            task_to_valid_loaders[task_name].dataset.transform = supervised_test_transform
            task_to_test_loaders[task_name].dataset.transform = supervised_test_transform
            evaluator = LinearEvaluation(model, task_to_train_loaders[task_name], task_to_valid_loaders[task_name], task_to_test_loaders[task_name],
                                        device, state_dict_dir=checkpoint_dir, state_dict_name="model_best")
            eval_log  = evaluator.eval()
            eval_log = {task_name + "_" + key: val for key, val in eval_log.items()}
            test_log.update(eval_log)

        for key, val in test_log.items():
            if key in test_metrics:
                test_metrics[key].append(val)
            else:
                test_metrics[key] = [val, ]

    end = time.time()
    logger.info("Total time: {:.4f} seconds".format((end - start)))

    # print training results
    for key, vals in test_metrics.items():
        logger.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(vals), np.std(vals)))

    # save results into .csv
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # save test results
    augment_str = "_".join(
                    [f"{name}_{ratio}" for (name, ratio) in zip(args.augmentation_names, args.augmentation_ratios)]
                )
    result_datapoint = {
        "Trained on": augment_str,
    }
    for key, vals in test_metrics.items():
        result_datapoint[key] = np.mean(vals)
        result_datapoint[key+"_std"] = np.std(vals)
    file_name = os.path.join(file_dir, "{}_test.csv".format(args.save_name))
    add_result_to_csv(result_datapoint, file_name)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--datasets', default=["Messidor", "Aptos", "Jinchi"], type=str, nargs='+',)
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, nargs='+',
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--use_valid', action='store_true')
    args.add_argument('--temperature', default=0.5, type=float)
    args.add_argument('--world_size', default=1, type=int)

    args.add_argument('--train_simclr', action='store_true')
    args.add_argument('--train_randaugment', action='store_true')
    args.add_argument('--train_no_transforms', action='store_true')
    args.add_argument('--train_bilevel', action='store_true')
    args.add_argument('--runs', type=int, default=3)
    args.add_argument('--randaugment_n', type=int, default=2)
    args.add_argument('--randaugment_m', type=int, default=10)

    args.add_argument('--augmentation_names', nargs = "+", type=str, default=["Identity"], help='augmentation names')
    args.add_argument('--augmentation_ratios', nargs = "+", type=float, default=[0.2], help='augmentation ratios')
    args.add_argument('--augmentation_probs', nargs = "+", type=float, default=[], help='augmentation probs')
    args.add_argument('--tree_idxes', nargs='+', type=int, default=[], help='tree index')

    args.add_argument('--img_size', type=int, default=224)
    args.add_argument("--vit_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args.add_argument("--vit_pretrained_dir", type=str, default="pretrained/imagenet21k_ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    args.add_argument('--save_name', type=str, default="test", help='save name')

    args.add_argument('--weight_lr', type=float, default=1)
    args.add_argument('--collect_gradient_step', type=float, default=1)
    args.add_argument('--update_weight_step', type=float, default=50)
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;args;encoder_name"),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--n_gpu'], type=int, target='n_gpu'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)