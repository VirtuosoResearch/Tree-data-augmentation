from trainer import Trainer
from utils import num_graphs
import torch

class SimCLRTrainer(Trainer):

    def __init__(self, model, optimizer, lr_scheduler, criterion, metric_ftns, 
                 train_loader, valid_loader, test_loader, device, logger, epochs, save_epochs, 
                 checkpoint_dir, mnt_metric="val_accuracy", mnt_mode="max"):
        super().__init__(model, optimizer, lr_scheduler, criterion, metric_ftns, 
            train_loader, valid_loader, test_loader, device, logger, epochs, save_epochs, 
            checkpoint_dir, mnt_metric, mnt_mode)


    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        self.train_metrics.reset()
        for data, data1, data2 in self.train_loader:
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
            metrics = self.valid_metrics
            data_loader = self.valid_loader
        else:
            metrics = self.test_metrics
            data_loader = self.test_loader

        metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, data1, data2) in enumerate(data_loader):
                if num_graphs(data1) == 1:
                    continue
                data1 = data1.to(self.device)
                data2 = data2.to(self.device)
                out1 = self.model.forward_cl(data1)
                out2 = self.model.forward_cl(data2)
                loss = self.criterion(out1, out2)

                metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    output = out1
                    target = data1.y.view(-1)
                    metrics.update(met.__name__, met(output, target).item(), n=num_graphs(data1))

        return metrics.result()