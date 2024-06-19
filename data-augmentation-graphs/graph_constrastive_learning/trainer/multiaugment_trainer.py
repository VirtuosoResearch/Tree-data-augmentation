from trainer.simclr_trainer import SimCLRTrainer
from utils import num_graphs
from torch.autograd import grad
import numpy as np
import json
import torch
import torch.nn.functional as F

class MultiaugmentSimCLRTrainer(SimCLRTrainer):

    def __init__(self, model, optimizer, lr_scheduler, criterion, metric_ftns,
                 train_loader, valid_loader, test_loader, device, logger, epochs,
                 save_epochs, checkpoint_dir, mnt_metric, mnt_mode,
                 multitask_train_dataloader):
        super().__init__(model, optimizer, lr_scheduler, criterion, metric_ftns, 
                         train_loader, valid_loader, test_loader, device, logger, epochs, 
                         save_epochs, checkpoint_dir, mnt_metric, mnt_mode)
        self.multitask_train_dataloader = multitask_train_dataloader

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        self.train_metrics.reset()
        train_step = 0
        for batch in self.multitask_train_dataloader:
            data, data1, data2 = batch['sample']
            if num_graphs(data1) == 1:
                continue
            self.optimizer.zero_grad()
            
            data1 = data1.to(self.device)
            data2 = data2.to(self.device)
            out1 = self.model.forward_cl(data1)
            out2 = self.model.forward_cl(data2)
            loss = self.criterion(out1, out2)
            loss.backward()
            self.optimizer.step()
            
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                output = out1
                target = data1.y.view(-1)
                self.train_metrics.update(met.__name__, met(output, target).item(), n=num_graphs(data1))

            # only conduct the same steps as the a single train_loader
            train_step += 1
            if train_step >= len(self.train_loader):
                break

        log = self.train_metrics.result()
        
        if self.valid_loader is not None:
            val_log = self.eval(phase="valid")
            log.update(**{'valid_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log


class TAGTrainer(SimCLRTrainer):
    
    def __init__(self, model, optimizer, lr_scheduler, criterion, metric_ftns,
                train_loader, valid_loader, test_loader, device, logger, epochs, save_epochs, 
                checkpoint_dir, mnt_metric, mnt_mode,
                multitask_train_dataloader, train_data_loaders,
                collect_gradient_step = 4, update_lr = 1e-3, record_step = 10, affinity_dir = '', affinity_type='tag', target_tasks = []):
        super().__init__(model, optimizer, lr_scheduler, criterion, metric_ftns, 
                train_loader, valid_loader, test_loader, device, logger, epochs, save_epochs, 
                checkpoint_dir, mnt_metric, mnt_mode)
        
        self.multitask_train_dataloader = multitask_train_dataloader
        self.train_data_loaders = train_data_loaders

        self.collect_gradient_step = collect_gradient_step
        self.update_lr = update_lr
        self.affinity_dir = affinity_dir
        self.affinity_type = affinity_type

        self.global_step = 0
        self.record_step = record_step

        tasks = list(self.train_data_loaders.keys())
        self.task_gains = {task: dict([(t, []) for t in tasks]) for task in tasks}
        self.target_tasks = target_tasks if len(target_tasks) > 0 else tasks

    def save_task_gains(self):
        for task, gains in self.task_gains.items():
            for other_task in gains.keys():
                gains[other_task] = np.mean(gains[other_task])
        
        # save the task affinity
        with open(self.affinity_dir, "w") as f:
            task_affinity = json.dumps(self.task_gains)
            f.write(task_affinity)
        self.logger.info("Saving TAG task affinity...")
        self.logger.info(task_affinity)

    def update_task_gains(self, step_gains):
        for task, task_step_gain in step_gains.items():
            for other_task in task_step_gain.keys():
                self.task_gains[task][other_task].append(task_step_gain[other_task])

    def compute_loss_and_gradients(self, data_loader, task_name, step=1):
        loss = 0; count=0
        
        for batch in data_loader:
            if count > step:
                break
            data, data1, data2 = batch
            if num_graphs(data) == 1:
                continue
            self.optimizer.zero_grad()
            
            data1 = data1.to(self.device)
            data2 = data2.to(self.device)
            out1 = self.model.forward_cl(data1)
            out2 = self.model.forward_cl(data2)
            batch_loss = self.criterion(out1, out2)
            
            loss += batch_loss
            count += 1

        loss = loss/count
        feature_gradients = grad(loss, self.model.parameters(), retain_graph=False, create_graph=False, allow_unused=True)
        return loss.cpu().item(), feature_gradients

    def compute_tag_task_gains(self):
        task_gain = {str(task): dict() for task in self.target_tasks}

        # 1. collect task losses
        task_losses = {}
        task_gradients = {}

        for task, train_data_loader in self.train_data_loaders.items():
            tmp_loss, tmp_gradients = self.compute_loss_and_gradients(train_data_loader, task, self.collect_gradient_step)

            task_losses[task] = tmp_loss
            task_gradients[task] = tmp_gradients
            
        
        for task in self.train_data_loaders.keys():
            # 2. take a gradient step on the task loss
            encoder_weights = list(self.model.parameters())
            encoder_gradients = task_gradients[task]
            for i, weight in enumerate(encoder_weights):
                if encoder_gradients[i] is None:
                    continue
                weight.data -= encoder_gradients[i].data * self.update_lr

            # 3. evaluate losses on the target task
            other_tasks = self.target_tasks
            for other_task in other_tasks:
                update_loss, _ = self.compute_loss_and_gradients(self.train_data_loaders[other_task], other_task, self.collect_gradient_step)
                task_gain[other_task][task] =  1 - update_loss/task_losses[other_task]

            # 4. restore weights
            for i, weight in enumerate(encoder_weights):
                if encoder_gradients[i] is None:
                    continue
                weight.data += encoder_gradients[i].data * self.update_lr

        return task_gain
    
    def compute_cs_task_gains(self):
        task_gain = {str(task): dict() for task in self.target_tasks}

        # 1. collect task losses and gradients
        task_losses = {}
        task_gradients = {}

        for task, train_data_loader in self.train_data_loaders.items():
            tmp_loss, tmp_gradients = self.compute_loss_and_gradients(train_data_loader, task, self.collect_gradient_step)
            tmp_gradients = torch.concat([gradient.view(-1) for gradient in tmp_gradients if gradient is not None])
            task_losses[task] = tmp_loss
            task_gradients[task] = tmp_gradients
    
        # 2. compute cosine similarity
        other_tasks =  self.target_tasks
        for other_task in other_tasks:
            for task in self.train_data_loaders.keys():
                task_gain[str(other_task)][str(task)] = F.cosine_similarity(
                    task_gradients[other_task], task_gradients[task], dim=0
                ).cpu().item()

        return task_gain

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        self.train_metrics.reset()
        train_step = 0
        for batch in self.multitask_train_dataloader:
            data, data1, data2 = batch['sample']
            if num_graphs(data1) == 1:
                continue
            self.optimizer.zero_grad()
            
            data1 = data1.to(self.device)
            data2 = data2.to(self.device)
            out1 = self.model.forward_cl(data1)
            out2 = self.model.forward_cl(data2)
            loss = self.criterion(out1, out2)
            loss.backward()
            self.optimizer.step()
            
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                output = out1
                target = data1.y.view(-1)
                self.train_metrics.update(met.__name__, met(output, target).item(), n=num_graphs(data1))

            # # only conduct the same steps as the a single train_loader
            # train_step += 1
            # if train_step >= len(self.train_loader):
            #     break

            # compute iter-task affinity
            if self.global_step % self.record_step == 0:
                if self.affinity_type == 'tag':
                    step_task_gains = self.compute_tag_task_gains()
                elif self.affinity_type == 'cs':
                    step_task_gains = self.compute_cs_task_gains()
                self.update_task_gains(step_task_gains)
            self.global_step += 1

        log = self.train_metrics.result()
        
        if self.valid_loader is not None:
            val_log = self.eval(phase="valid")
            log.update(**{'valid_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log