from trainer.simclr_trainer import SimCLRTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tawt import get_task_weights_gradients_multi
from utils import num_graphs

class BilevelTrainer(SimCLRTrainer):

    def __init__(self, model, optimizer, lr_scheduler, criterion, metric_ftns, 
                 train_loader, valid_loader, test_loader, device, logger, epochs, 
                 save_epochs, checkpoint_dir, mnt_metric, mnt_mode,
                 multitask_train_dataloader, train_data_loaders, weight_lr=1, collect_gradient_step=1):
        super().__init__(model, optimizer, lr_scheduler, criterion, metric_ftns, 
                         train_loader, valid_loader, test_loader, device, logger, epochs, 
                         save_epochs, checkpoint_dir, mnt_metric, mnt_mode)
        
        self.multitask_train_dataloader = multitask_train_dataloader
        self.train_data_loaders = train_data_loaders

        self.task_list = list(train_data_loaders.keys())
        self.num_tasks = len(self.task_list)
        self.task_to_index = dict([(task, i) for i, task in enumerate(self.task_list)])

        self.weight_lr = weight_lr
        self.task_weights = torch.ones((self.num_tasks, ), device=self.device)/self.num_tasks
        self.collect_gradient_step = collect_gradient_step
        
    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        self.train_metrics.reset()
        for batch in self.multitask_train_dataloader:
            task_name = batch['task_name'][0]
            data, data1, data2 = batch['sample']
            if num_graphs(data1) == 1:
                continue
            self.optimizer.zero_grad()
            
            data1 = data1.to(self.device)
            data2 = data2.to(self.device)
            out1 = self.model.forward_cl(data1)
            out2 = self.model.forward_cl(data2)
            loss = self.criterion(out1, out2)
            loss = loss*self.task_weights[self.task_to_index[task_name]]*self.num_tasks
            loss.backward()
            self.optimizer.step()
            
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                output = out1
                target = data1.y.view(-1)
                self.train_metrics.update(met.__name__, met(output, target).item(), n=num_graphs(data1))

        '''
        Update task weights every epoch
        '''
        task_weights_gradients = get_task_weights_gradients_multi(
            self.model, self.train_data_loaders,
            self.criterion, self.device, self.collect_gradient_step
        )
        exp_ratio = torch.exp(- self.weight_lr * task_weights_gradients)
        new_task_weights = self.task_weights * exp_ratio
        self.task_weights = new_task_weights/torch.sum(new_task_weights)
        self.logger.info(self.task_list)
        self.logger.info(self.task_weights)

        log = self.train_metrics.result()
        
        if self.valid_loader is not None:
            val_log = self.eval(phase="valid")
            log.update(**{'valid_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log