import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from numpy import inf

from utils import num_graphs, MetricTracker

class Trainer:
    '''
    Basic logic for graph prediction learning 
    '''
    def __init__(self, model, optimizer, lr_scheduler, criterion, metric_ftns, 
                train_loader, valid_loader, test_loader, device, logger,
                epochs, save_epochs, checkpoint_dir, mnt_metric="val_accuracy", mnt_mode = "max"):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device

        ''' Training config '''
        self.epochs = epochs
        self.save_epochs = save_epochs
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # configuration to monitor model performance and save best

        self.mnt_mode, self.mnt_metric = mnt_mode, mnt_metric
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics  = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.save_checkpoint(0)

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        self.train_metrics.reset()
        for data in self.train_loader:
            self.optimizer.zero_grad()
            
            data = data.to(self.device)
            target = data.y.view(-1)
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target).item(), n=num_graphs(data))

        log = self.train_metrics.result()
        
        if self.valid_loader is not None:
            val_log = self.eval(phase="valid")
            log.update(**{'valid_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log
    
    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(1, self.epochs + 1):
            result = self.train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                # if not_improved_count > self.early_stop:
                #     self.logger.info("Validation performance didn\'t improve for {} epochs. "
                #                      "Training stops.".format(self.early_stop))
                #     break

            if best:
                self.save_checkpoint(epoch)
            if epoch % self.save_epochs == 0:
                self.save_checkpoint(epoch, name=f'model_epoch_{epoch}')

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
            for batch_idx, data in enumerate(data_loader):
                data = data.to(self.device)
                target = data.y.view(-1)

                output = self.model(data)
                loss = self.criterion(output, target)

                metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    metrics.update(met.__name__, met(output, target).item(), n=num_graphs(data))

        return metrics.result()

    def test(self, phase="test"):
        # load model checkpoint
        model_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        self.model.load_state_dict(torch.load(model_path))
        self.logger.info(f"Loading model: {model_path} ...")

        log = {}
        test_log = self.eval(phase=phase)
        log.update(**{f'{phase}_'+k : v for k, v in test_log.items()})

        return log
    
    def save_checkpoint(self, epoch, name = "model_best"):
        model_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Saving current model: {name}.pth ...")