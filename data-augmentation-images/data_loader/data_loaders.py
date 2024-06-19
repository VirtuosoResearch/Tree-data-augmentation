from torchvision import datasets, transforms
from .base_data_loader import BaseDataLoader, DataLoader
from .dataset_cifar import CIFAR10, CIFAR100
import transforms 
from .multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from torch.utils.data.dataloader import default_collate
from data_loader.messidor2 import Messidor2
from data_loader.jinchi import Jinchi
from data_loader.aptos import Aptos

name_to_datasets = {
    "Cifar10DataLoader": CIFAR10,
    "Cifar100DataLoader": CIFAR100,
}

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers=num_workers)

class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, num_workers=1, phase="train", transform=None):
        training = phase == "train"
        
        trsfm = transform # getattr(transforms, transform_name)()
        self.data_dir = data_dir
        self.dataset = CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class Cifar100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, num_workers=1, phase="train", transform=None):
        training = phase == "train"
        
        trsfm = transform # getattr(transforms, transform_name)()
        self.data_dir = data_dir
        self.dataset = CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class MessidorDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, test_split=0.0, num_workers=1, phase="train", transform=None):
        training = phase == "train"
        assert transform is not None
        
        self.data_dir = data_dir
        self.dataset = Messidor2(self.data_dir, train=training, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=test_split, num_workers=num_workers)

class AptosDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, test_split=0.0, num_workers=1, phase="train", transform=None):
        training = phase == "train"
        assert transform is not None
        
        self.data_dir = data_dir
        self.dataset = Aptos(self.data_dir, train=training, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=test_split, num_workers=num_workers)

class JinchiDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, test_split=0.0, num_workers=1, phase="train", transform=None):
        training = phase == "train"
        assert transform is not None
        
        self.data_dir = data_dir
        self.dataset = Jinchi(self.data_dir, train=training, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=test_split, num_workers=num_workers)

def load_multiaugment_dataloader(dataset_name, data_dir, batch_size, transform_names):
    task_to_train_datasets = {}
    for name in transform_names:
        name = name.split("_")
        if len(name) == 1:
            name = name[0]
            id = 0
        else:
            assert len(name) == 2
            name, id = name[0], int(name[1])
        task_name = name + f"_{id}"
        task_to_train_datasets.update({
                task_name: name_to_datasets[dataset_name](data_dir, train=True, 
                download=True, transform=getattr(transforms, name)(n=10, seed=id)) 
            })

    task_to_collator = {
        key: default_collate for key in task_to_train_datasets.keys()
    }

    task_to_train_dataloaders = {
        key: BaseDataLoader(task_to_train_datasets[key], batch_size, valid_split = 0.1, shuffle=True, num_workers=2) \
        for key in task_to_train_datasets.keys()
    }

    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, batch_size)
    multitask_train_collator = MultitaskCollator(task_to_collator)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
        collate_fn=multitask_train_collator.collator_fn,
    )
    return multitask_train_dataloader, task_to_train_dataloaders

def load_multitask_dataloader(task_to_train_datasets, batch_size):
    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, batch_size)
    task_to_collator = {
        key: default_collate for key in task_to_train_datasets.keys()
    }
    multitask_train_collator = MultitaskCollator(task_to_collator)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
        collate_fn=multitask_train_collator.collator_fn,
    )
    return multitask_train_dataloader