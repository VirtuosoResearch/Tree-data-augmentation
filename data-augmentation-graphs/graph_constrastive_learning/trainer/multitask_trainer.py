from trainer.simclr_trainer import SimCLRTrainer
from utils import num_graphs, MetricTracker
import torch

class MultitaskSimCLRTrainer(SimCLRTrainer):

    def __init__(self, model, optimizer, lr_scheduler, criterion, metric_ftns,
                 train_loader, valid_loader, test_loader, device, logger, epochs,
                 save_epochs, checkpoint_dir, mnt_metric, mnt_mode,
                 multitask_train_dataloader, valid_loaders, test_loaders):
        super().__init__(model, optimizer, lr_scheduler, criterion, metric_ftns, 
                         train_loader, valid_loader, test_loader, device, logger, epochs, 
                         save_epochs, checkpoint_dir, mnt_metric, mnt_mode)
        self.multitask_train_dataloader = multitask_train_dataloader
        self.valid_loaders = valid_loaders
        self.test_loaders = test_loaders

        self.task_to_valid_metrics = {task: MetricTracker('loss', *[m.__name__ for m in self.metric_ftns]) for task in self.valid_loaders.keys()}
        self.task_to_test_metrics  = {task: MetricTracker('loss', *[m.__name__ for m in self.metric_ftns]) for task in self.test_loaders.keys()}

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
            out1 = self.model.forward_cl(task_name, data1)
            out2 = self.model.forward_cl(task_name, data2)
            loss = self.criterion(out1, out2)
            loss.backward()
            self.optimizer.step()
            
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                output = out1
                target = data1.y.view(-1)
                self.train_metrics.update(met.__name__, met(output, target).item(), n=num_graphs(data1))

        log = self.train_metrics.result()
        
        if self.valid_loader is not None:
            val_log = self.eval(phase="valid")
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log
    
    def eval(self, phase = "valid"):
        self.model.eval()

        if phase == "valid":
            task_metrics = self.task_to_valid_metrics
            task_data_loaders = self.valid_loaders
        else:
            task_metrics = self.task_to_test_metrics
            task_data_loaders = self.test_loaders

        log = {}
        avg_loss = 0
        for task_name in task_data_loaders.keys():
            data_loader = task_data_loaders[task_name]
            metrics = task_metrics[task_name]
            metrics.reset()
            with torch.no_grad():
                for batch_idx, (data, data1, data2) in enumerate(data_loader):
                    if num_graphs(data1) == 1:
                        continue
                    data1 = data1.to(self.device)
                    data2 = data2.to(self.device)
                    out1 = self.model.forward_cl(task_name, data1)
                    out2 = self.model.forward_cl(task_name, data2)
                    loss = self.criterion(out1, out2)

                    metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        output = out1
                        target = data1.y.view(-1)
                        metrics.update(met.__name__, met(output, target).item(), n=num_graphs(data1))

            task_log = metrics.result() 
            avg_loss += task_log['loss']
            task_log = {f"{task_name}_{key}": val for key, val in task_log.items()}
            log.update(**task_log)
        log.update(**{'loss': avg_loss/len(task_data_loaders.keys())})
        return log