from abc import abstractmethod

import os
import numpy as np
import torch
from numpy import inf
from utils import MetricTracker, inf_loop
import time

class Trainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config,
                 device,
                 train_data_loader, 
                 valid_data_loader=None,
                 test_data_loader=None, 
                 lr_scheduler=None,
                 checkpoint_dir=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.device = device

        self.model = model.to(device)
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is None:
            self.checkpoint_dir = config.save_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # else:
        #     for filename in os.listdir(self.checkpoint_dir):
        #         os.remove(os.path.join(self.checkpoint_dir, filename))

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.len_epoch = len(self.train_data_loader)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        # self._save_checkpoint(0, name="model_epoch_0.pth")


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, index) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

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

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if best:
                self._save_checkpoint(epoch)
            if epoch % 50 == 0:
                self._save_checkpoint(epoch, name=f'model_epoch_{epoch}.pth')

    def test(self, use_val = False):
        best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
        if os.path.exists(best_path):
            first_key = torch.load(best_path, map_location=self.device)["state_dict"].keys().__iter__().__next__()
            if (hasattr(self.model, 'module') and first_key.startswith('module')):
                self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
            elif not hasattr(self.model, 'module'):
                new_state_dict = {k.replace('module.', ''): v for k, v in torch.load(best_path, map_location=self.device)["state_dict"].items()}
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.module.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])

        self.model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))
        data_loader = self.valid_data_loader if use_val else self.test_data_loader
        with torch.no_grad():
            for i, (data, target, index) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # computing loss, metrics on test set
                loss = self.criterion(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_ftns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })
        self.logger.info(log)

        if use_val:
            return {'val_'+k : v for k, v in log.items()}
        else:
            return {'test_'+k : v for k, v in log.items()}

    def _save_checkpoint(self, epoch, name = 'model_best.pth'):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        if hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict
        }
        self.logger.info("Best checkpoint in epoch {}".format(epoch))
        
        best_path = os.path.join(self.checkpoint_dir, name)
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")

    def val(self):
        best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
        if os.path.exists(best_path):
            first_key = torch.load(best_path, map_location=self.device)["state_dict"].keys().__iter__().__next__()
            if (hasattr(self.model, 'module') and first_key.startswith('module')):
                self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
            elif not hasattr(self.model, 'module'):
                new_state_dict = {k.replace('module.', ''): v for k, v in torch.load(best_path, map_location=self.device)["state_dict"].items()}
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.module.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])

        self.model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))

        with torch.no_grad():
            for i, (data, target, index) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # computing loss, metrics on test set
                loss = self.criterion(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_ftns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(self.valid_data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })
        self.logger.info(log)

        return log["accuracy"]
    

from utils.tawt import get_task_weights_gradients_multi
import torch.nn.functional as F

class BilevelSupervisedTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
                train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir,
                train_data_loaders, weight_lr=1, collect_gradient_step=1, update_weight_step = 50):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
                train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        
        self.train_data_loaders = train_data_loaders

        self.task_list = list(train_data_loaders.keys())
        self.num_tasks = len(self.task_list)
        self.task_list = [int(task_name) for task_name in self.task_list]

        self.weight_lr = weight_lr
        self.task_weights = torch.ones((self.num_tasks, ), device=self.device)/self.num_tasks
        self.collect_gradient_step = collect_gradient_step
        self.update_weight_step = update_weight_step

    def convert_idx(self, group_idx):
        for i, task_name in enumerate(self.task_list):
            group_idx[group_idx == task_name] = i
        return group_idx

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_step = 1
        for batch_idx, (data, target, meta_data) in enumerate(self.train_data_loader):
            data, target, meta_data = data.to(self.device), target.to(self.device), meta_data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target, reduce='none')
            group_idx = self.convert_idx(meta_data[:, 0]).type(torch.long)
            group_weights = self.task_weights[group_idx]
            loss = (loss * group_weights * self.num_tasks).sum()

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break

            '''
            Update task weights
            '''
            if train_step % self.update_weight_step == 0:
                task_weights_gradients = get_task_weights_gradients_multi(
                    self.model, self.train_data_loaders,
                    self.criterion, self.device, self.collect_gradient_step, if_supervised = True
                )
                exp_ratio = torch.exp(- self.weight_lr * task_weights_gradients)
                new_task_weights = self.task_weights * exp_ratio
                self.task_weights = new_task_weights/torch.sum(new_task_weights)
                self.logger.info(self.task_weights)

            train_step += 1


        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
class DROSupervisedTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
                train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir,
                train_data_loaders, weight_lr=1, collect_gradient_step=1, update_weight_step = 50):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
                train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        
        self.train_data_loaders = train_data_loaders

        self.task_list = list(train_data_loaders.keys())
        self.num_tasks = len(self.task_list)
        self.task_list = [int(task_name) for task_name in self.task_list]

        self.weight_lr = weight_lr
        self.task_weights = torch.ones((self.num_tasks, ), device=self.device)/self.num_tasks
        self.collect_gradient_step = collect_gradient_step
        self.update_weight_step = update_weight_step

    def convert_idx(self, group_idx):
        for i, task_name in enumerate(self.task_list):
            group_idx[group_idx == task_name] = i
        return group_idx

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_step = 1; task_loss = torch.zeros(self.num_tasks).to(self.device)
        for batch_idx, (data, target, meta_data) in enumerate(self.train_data_loader):
            data, target, meta_data = data.to(self.device), target.to(self.device), meta_data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target, reduce='none')
            group_idx = self.convert_idx(meta_data[:, 0]).type(torch.long)
            group_weights = self.task_weights[group_idx]
            sample_losses = loss * group_weights * self.num_tasks
            task_loss.index_add_(0, group_idx, sample_losses.detach().clone())

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break

            '''
            Update task weights
            '''
            if train_step % self.update_weight_step == 0:
                avg_task_loss = task_loss/(batch_idx+1)
                exp_ratio = torch.exp(self.weight_lr * avg_task_loss)
                new_task_weights = self.task_weights * exp_ratio
                self.task_weights = new_task_weights/torch.sum(new_task_weights)
                self.logger.info(self.task_weights)

            train_step += 1


        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
