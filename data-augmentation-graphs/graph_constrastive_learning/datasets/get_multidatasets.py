from datasets.multitask_dataset import MultitaskDataset, MultitaskBatchSampler
from torch_geometric.loader import DataLoader
import transforms

def load_multiaugment_dataloader(dataset, batch_size, transform_names, transform_ratios):
    task_to_train_datasets = {}
    
    for name, ratio in zip(transform_names, transform_ratios):
        tmp_dataset = dataset.copy()
        tmp_dataset.transform = transforms.SimCLRTransfrom(
            transform_names=[name], 
            ratios=[ratio])
        task_to_train_datasets.update({
                f"{name}_{ratio}": tmp_dataset
            })

    task_to_train_dataloaders = {
        key: DataLoader(task_to_train_datasets[key], batch_size, shuffle=True) \
        for key in task_to_train_datasets.keys()
    }

    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, batch_size)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
    )
    return multitask_train_dataloader, task_to_train_dataloaders

def load_multiaugment_dataloader_v2(dataset, batch_size, transform_names, transform_ratios):
    '''Deprecated'''
    task_to_train_datasets = {}
    transform_name_combos = []
    # combine every two transforms
    for i, name1 in enumerate(transform_names):
        for j, name2 in enumerate(transform_names):
            if j>=i:
                transform_name_combos.append((name1, name2, transform_ratios[i], transform_ratios[j]))

    for transform_comb in transform_name_combos:
        name1, name2, ratio1, ratio2 = transform_comb
        tmp_dataset = dataset.copy()
        tmp_dataset.transform = transforms.SimCLRTransfrom_v2(
            transform_names_1=[name1], transform_names_2=[name2], 
            ratios_1=[ratio1], ratios_2=[ratio2])
        task_to_train_datasets.update({
                f"{name1}_{name2}_{ratio1}_{ratio2}": tmp_dataset
            })

    task_to_train_dataloaders = {
        key: DataLoader(task_to_train_datasets[key], batch_size, shuffle=True) \
        for key in task_to_train_datasets.keys()
    }

    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, batch_size)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
    )
    return multitask_train_dataloader, task_to_train_dataloaders

def load_multitask_dataloader(task_to_train_datasets, batch_size):
    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, batch_size)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
    )
    return multitask_train_dataloader