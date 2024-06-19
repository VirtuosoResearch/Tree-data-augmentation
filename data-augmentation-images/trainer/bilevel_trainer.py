from trainer.simclr_trainer import SimCLRTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tawt import get_task_weights_gradients_multi

class BilevelTrainer(SimCLRTrainer):
    
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
                 train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir, 
                 multitask_train_dataloader, train_data_loaders,
                 weight_lr=1, collect_gradient_step=1, update_weight_step = 50):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        
        self.multitask_train_dataloader = multitask_train_dataloader
        self.train_data_loaders = train_data_loaders

        self.task_list = list(train_data_loaders.keys())
        self.num_tasks = len(self.task_list)
        self.task_to_index = dict([(task, i) for i, task in enumerate(self.task_list)])

        self.weight_lr = weight_lr
        self.task_weights = torch.ones((self.num_tasks, ), device=self.device)/self.num_tasks
        self.collect_gradient_step = collect_gradient_step
        self.update_weight_step = update_weight_step

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_step = 1
        for batch_idx, batch in enumerate(self.multitask_train_dataloader):
            task_name = batch['task_name']
            data, target, index  = batch['data']
            data_1, data_2 = data
            data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

            self.optimizer.zero_grad()
            h_i, h_j, z_i, z_j = self.model(task_name, data_1, data_2)
            output = h_i

            loss = self.criterion(z_i, z_j)
            loss = loss*self.task_weights[self.task_to_index[task_name]]*self.num_tasks
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
            
            # only conduct the same steps as the a single train_loader
            train_step += 1
            
            if train_step >= len(self.train_data_loader):
                break

            '''
            Update task weights
            '''
            if train_step % self.update_weight_step == 0:
                task_weights_gradients = get_task_weights_gradients_multi(
                    self.model, self.train_data_loaders,
                    self.criterion, self.device, self.collect_gradient_step
                )
                exp_ratio = torch.exp(- self.weight_lr * task_weights_gradients)
                new_task_weights = self.task_weights * exp_ratio
                self.task_weights = new_task_weights/torch.sum(new_task_weights)
                self.logger.info(self.task_weights)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log