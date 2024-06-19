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
from transforms.compositions import SimCLRTransfrom, SimCLR
from transforms.simclr_augment import SimCLRTestTransfrom
from transforms.rand_augment import RandAugment
from transforms.basic import TestTransform, CIFARTestTransform
import time

def main(config, args):
    start = time.time()
    logger = config.get_logger('train')

    # setup data_loader instances
    if config['data_loader']['type'] == "MessidorDataLoader":
        pre_transforms = [
                tfrms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                tfrms.Pad((0, 37, 0, 38)),
                tfrms.Lambda(lambda x: x.convert("RGB")),
            ]
        post_transforms = [tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613])]
    elif config['data_loader']['type'] == "JinchiDataLoader":
        pre_transforms = [
                tfrms.Resize((224, 224)), 
                tfrms.Lambda(lambda x: x.convert("RGB")),
            ]
        post_transforms = [tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613])]
    elif config['data_loader']['type'] == "AptosDataLoader":
        pre_transforms = [
                tfrms.Resize(224), 
                tfrms.Lambda(lambda x: x.convert("RGB")),
            ]
        post_transforms = [tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613])]
    elif config['data_loader']['type'] == "Cifar10DataLoader" or config['data_loader']['type'] == "Cifar100DataLoader":
        pre_transforms = []; post_transforms = []
    else:
        print("Unknown data loader type!")

    augmentation_probs = args.augmentation_probs if len(args.augmentation_probs) > 0 else None
    tree_idxes = args.tree_idxes if len(args.tree_idxes) > 0 else None

    size = 32 if config['data_loader']['type'] == "Cifar10DataLoader" else 224
    if args.train_simclr:
        transform = SimCLR(pre_transforms=pre_transforms, post_transforms=post_transforms, size = size)
    elif args.train_randaugment:
        pre_transforms.append(RandAugment(args.randaugment_n, args.randaugment_m))
        pre_transforms += [
                tfrms.RandomResizedCrop((size, size)),
                tfrms.RandomHorizontalFlip(),
            ]
        transform = SimCLRTransfrom(pre_transforms=pre_transforms, post_transforms=post_transforms)
    elif args.test_simclr:
        pre_transforms += [
                tfrms.RandomResizedCrop((size, size)),
                tfrms.RandomHorizontalFlip(),
            ]
        transform = SimCLRTestTransfrom(transform_names=args.augmentation_names, ratios=args.augmentation_ratios, 
                                        pre_transforms=pre_transforms, post_transforms=post_transforms,
                                        probs=augmentation_probs, tree_idxes=tree_idxes)
    else:
        pre_transforms += [
                tfrms.RandomResizedCrop((size, size)),
                tfrms.RandomHorizontalFlip(),
            ]
        transform = SimCLRTransfrom(transform_names=args.augmentation_names, ratios=args.augmentation_ratios, 
                                    probs=augmentation_probs, tree_idxes=tree_idxes, 
                                    pre_transforms=pre_transforms, post_transforms=post_transforms)

    if config['data_loader']['type'] == "MessidorDataLoader" or config['data_loader']['type'] == "JinchiDataLoader" or config['data_loader']['type'] == "AptosDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.2, test_split=0.2, phase = "train", transform=transform)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    elif config['data_loader']['type'] == "Cifar10DataLoader" or config['data_loader']['type'] == "Cifar100DataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train", valid_split = 0.1, transform=transform)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test", transform=transform)

    assert 0 < args.data_frac <= 1
    if args.downsample_dataset > -1:
        train_data_len = len(train_data_loader.sampler)
        assert 0 < args.downsample_dataset < train_data_len
        train_data_loader.sampler.indices = train_data_loader.sampler.indices[:args.downsample_dataset]
    elif args.data_frac < 1:
        train_data_len = len(train_data_loader.sampler)
        train_data_loader.sampler.indices = train_data_loader.sampler.indices[:int(train_data_len*args.data_frac)]
    logger.info("Train Size: {} Valid Size: {} Test Size: {}".format(
        len(train_data_loader.sampler), 
        len(valid_data_loader.sampler), 
        len(test_data_loader.sampler)))

    model = config.init_obj('arch', module_arch, pretrained = args.pretrained,
                            vit_type=args.vit_type, img_size=args.img_size, vit_pretrained_dir=args.vit_pretrained_dir)
    logger.info(model)

    test_metrics = {}

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if args.load_model_dir != "":
        load_model_dir = f"./saved/models/{args.load_model_dir}"
        model.load_state_dict(torch.load(load_model_dir, map_location=device)["state_dict"])

    # get function handles of loss and metrics
    criterion = module_loss.NT_Xent(config["data_loader"]["args"]["batch_size"], args.temperature, args.world_size)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    augment_str = "_".join(
                    [f"{name}_{ratio}" for (name, ratio) in zip(args.augmentation_names, args.augmentation_ratios)]
                )
    checkpoint_dir = "./saved/models/simclr_{}_{}/".format(
        config["arch"]["args"]["encoder_name"], 
        augment_str if not args.train_randaugment else "randaugment") 


    if config['data_loader']['type'] == "MessidorDataLoader":
        supervised_train_transform = tfrms.Compose([
                    tfrms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                    tfrms.Pad((0, 37, 0, 38)),
                    tfrms.Lambda(lambda x: x.convert("RGB")),
                    tfrms.ToTensor(),
                    tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
        supervised_test_transform = tfrms.Compose([
                    tfrms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                    tfrms.Pad((0, 37, 0, 38)),
                    tfrms.Lambda(lambda x: x.convert("RGB")),
                    tfrms.ToTensor(),
                    tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
    elif config['data_loader']['type'] == "JinchiDataLoader":
        supervised_train_transform = tfrms.Compose([
                    tfrms.Resize((224, 224)), 
                    tfrms.Lambda(lambda x: x.convert("RGB")),
                    tfrms.ToTensor(),
                    tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
        supervised_test_transform = tfrms.Compose([
                    tfrms.Resize((224, 224)), 
                    tfrms.Lambda(lambda x: x.convert("RGB")),
                    tfrms.ToTensor(),
                    tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
    elif config['data_loader']['type'] == "AptosDataLoader":
        supervised_train_transform = tfrms.Compose([
                    tfrms.Resize(224), 
                    tfrms.Lambda(lambda x: x.convert("RGB")),
                    tfrms.ToTensor(),
                    tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
        supervised_test_transform = tfrms.Compose([
                    tfrms.Resize(224), 
                    tfrms.Lambda(lambda x: x.convert("RGB")),
                    tfrms.ToTensor(),
                    tfrms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
    elif config['data_loader']['type'] == "Cifar10DataLoader" or config['data_loader']['type'] == "Cifar100DataLoader":
        supervised_train_transform = tfrms.Compose([tfrms.ToTensor()])
        supervised_test_transform = tfrms.Compose([tfrms.ToTensor()])

    trainer = SimCLRTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      checkpoint_dir=checkpoint_dir)

    trainer.train()
    test_log = trainer.test(load_best=(config["trainer"]["epochs"]>0))
    
    train_data_loader.dataset.transform = supervised_train_transform
    valid_data_loader.dataset.transform = supervised_test_transform
    test_data_loader.dataset.transform = supervised_test_transform

    evaluator = LinearEvaluation(model, train_data_loader, valid_data_loader, test_data_loader,
                                device, state_dict_dir=checkpoint_dir, state_dict_name="model_best")
    for run in range(args.runs):
        eval_log  = evaluator.eval()
        test_log.update(eval_log)

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

    args.add_argument('--train_simclr', action='store_true')
    args.add_argument('--train_randaugment', action='store_true')
    args.add_argument('--randaugment_n', type=int, default=2)
    args.add_argument('--randaugment_m', type=int, default=10)

    args.add_argument('--test_simclr', action='store_true')
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

    args.add_argument("--load_model_dir", type=str, default="", help="Load model dir")
    args.add_argument("--pretrained", action="store_true", help="Load pretrained model")


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
