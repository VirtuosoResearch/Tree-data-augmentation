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
from utils import prepare_device, add_result_to_csv
from utils.linear_evaluation import LinearEvaluation
from torchvision import transforms as tfrms
from transforms.compositions import CompositeTransform, SimCLRTransfrom, SimCLR
from transforms.simclr_augment import SimCLRTestTransfrom
from transforms.rand_augment import RandAugment
from transforms.basic import TestTransform
import time

def main(config, args):
    logger = config.get_logger('train')

    augmentation_probs = args.augmentation_probs if len(args.augmentation_probs) > 0 else None
    tree_idxes = args.tree_idxes if len(args.tree_idxes) > 0 else None

    pre_transforms = [
            # tfrms.RandomCrop(32, padding=4),
            # tfrms.RandomHorizontalFlip(),
        ]
    post_transforms = [
        tfrms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ]
    transform = CompositeTransform(transform_names=args.augmentation_names, ratios=args.augmentation_ratios, 
                                probs=augmentation_probs, tree_idxes=tree_idxes, 
                                pre_transforms=pre_transforms, post_transforms=post_transforms)

    if config['data_loader']['type'] == "Cifar10DataLoader" or config['data_loader']['type'] == "Cifar100DataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train", valid_split = 0.1, transform=transform)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test", valid_split = 0.001, transform=transform)
    else:
        print("Unknown data loader type!")

    test_data_loader.dataset.transform = tfrms.Compose([tfrms.ToTensor(),
                            tfrms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])

    assert 0 < args.data_frac <= 1
    if args.data_frac < 1:
        train_data_len = len(train_data_loader.sampler)
        train_data_loader.sampler.indices = train_data_loader.sampler.indices[:int(train_data_len*args.data_frac)]
        valid_data_len = len(valid_data_loader.sampler)
        valid_data_loader.sampler.indices = valid_data_loader.sampler.indices[:int(valid_data_len*args.data_frac)]
        test_data_len = len(test_data_loader.sampler)
        test_data_loader.sampler.indices = test_data_loader.sampler.indices[:int(test_data_len*args.data_frac)]
    logger.info("Train Size: {} Valid Size: {} Test Size: {}".format(
        len(train_data_loader.sampler), 
        len(valid_data_loader.sampler), 
        len(test_data_loader.sampler)))
    
    augment_str = "_".join(
                    [f"{name}_{ratio}" for (name, ratio) in zip(args.augmentation_names, args.augmentation_ratios)]
                )
    load_model_dir = os.path.join("./saved/models", args.load_model_dir) \
        if args.load_model_dir is not None else None
    
    start = time.time()
    test_metrics = {}
    for run in range(args.runs):
        model = config.init_obj('arch', module_arch, vit_type=args.vit_type, img_size=args.img_size, vit_pretrained_dir=args.vit_pretrained_dir)
        if run == 0: logger.info(model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        if load_model_dir and os.path.exists(os.path.join(load_model_dir, "model_best.pth")):
            model.load_state_dict(
                torch.load(os.path.join(load_model_dir, "model_best.pth"))["state_dict"]
            )
        

        # get function handles of loss and metrics
        criterion = module_loss.nll_loss
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        checkpoint_dir = "./saved/models/supervised_training_{}_{}/".format(
            config["arch"]["type"], augment_str
        ) 

        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader,
                        test_data_loader=test_data_loader,
                        lr_scheduler=lr_scheduler,
                        checkpoint_dir=checkpoint_dir
                        )

        trainer.train()
        val_log = trainer.test(use_val=True)
        test_log = trainer.test()
        test_log.update(val_log)

        for key, val in test_log.items():
            if key in test_metrics:
                test_metrics[key].append(val)
            else:
                test_metrics[key] = [val, ]

    # print training results
    for key, vals in test_metrics.items():
        logger.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(vals), np.std(vals)))
    
    end = time.time()
    logger.info("Total Time: {:.4f}".format(end-start))

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

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, nargs='+',
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--use_valid', action='store_true')
    args.add_argument('--temperature', default=0.5, type=float)
    args.add_argument('--world_size', default=1, type=int)
    args.add_argument('--data_frac', type=float, default=1.0)
    args.add_argument('--downsample_dataset', type=int, default=-1)

    args.add_argument('--augmentation_names', nargs = "+", type=str, default=["Identity"], help='augmentation names')
    args.add_argument('--augmentation_ratios', nargs = "+", type=float, default=[0.2], help='augmentation ratios')
    args.add_argument('--augmentation_probs', nargs = "+", type=float, default=[], help='augmentation probs')
    args.add_argument('--tree_idxes', nargs = "+", type=int, default=[], help='tree idxes')

    args.add_argument('--img_size', type=int, default=224)
    args.add_argument("--vit_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args.add_argument("--vit_pretrained_dir", type=str, default="pretrained/imagenet21k_ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    args.add_argument('--save_name', type=str, default="test", help='save name')
    args.add_argument('--runs', type=int, default=3, help='number of runs')

    args.add_argument('--load_model_dir', type=str, default=None, help='load model dir')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;type"),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--n_gpu'], type=int, target='n_gpu'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
