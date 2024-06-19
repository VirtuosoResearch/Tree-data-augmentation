from trainer.simclr_trainer import SimCLRTrainer
import os
import torch

from utils import MetricTracker
from trainer.base_trainer import Trainer

class MultitaskSimCLRTrainer(SimCLRTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir,
            multitask_train_dataloader, train_data_loaders, valid_data_loaders, test_data_loaders):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        
        self.multitask_train_dataloader = multitask_train_dataloader
        self.train_data_loaders = train_data_loaders
        self.valid_data_loaders = valid_data_loaders
        self.test_data_loaders = test_data_loaders

        # self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        metric_names = ['loss', *[m.__name__ for m in self.metric_ftns]]
        metric_names = [f'{task_name}_{metric_name}' for task_name in self.valid_data_loaders.keys() for metric_name in metric_names]
        self.valid_metrics = MetricTracker(*metric_names)
    

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.multitask_train_dataloader):
            task_name = batch['task_name']
            data, target, index  = batch['data']

            data_1, data_2 = data
            data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

            self.optimizer.zero_grad()
            h_i, h_j, z_i, z_j = self.model(task_name, data_1, data_2)
            output = h_i

            loss = self.criterion(z_i, z_j)
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
        avg_loss = 0; batch_counts = 0
        with torch.no_grad():
            for task_name, data_loader in self.valid_data_loaders.items():
                for batch_idx, ((data_1, data_2), target, index) in enumerate(data_loader):
                    data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

                    h_i, h_j, z_i, z_j = self.model(task_name, data_1, data_2)
                    loss = self.criterion(z_i, z_j)
                    output = h_i

                    avg_loss += loss.item()
                    batch_counts += 1

                    self.valid_metrics.update(f'{task_name}_loss', loss.item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update(task_name + "_" + met.__name__, met(output, target))
        valid_log = self.valid_metrics.result()
        valid_log['loss'] = avg_loss / batch_counts

        return valid_log
    
    def test(self):
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
        self.model = self.model.to(self.device)
        self.model.eval()

        test_log = {}
        with torch.no_grad():
            for task_name, data_loader in self.test_data_loaders.items():
                total_loss = 0.0
                total_metrics = torch.zeros(len(self.metric_ftns))
                for i, ((data_1, data_2), target, index) in enumerate(data_loader):
                    data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

                    h_i, h_j, z_i, z_j = self.model(task_name, data_1, data_2)
                    output = h_i

                    # computing loss, metrics on test set
                    loss = self.criterion(z_i, z_j)
                    batch_size = data_1.shape[0]
                    total_loss += loss.item() * batch_size
                    for i, metric in enumerate(self.metric_ftns):
                        total_metrics[i] += metric(output, target) * batch_size

                n_samples = len(self.test_data_loader.sampler)
                task_log = {f'{task_name}_loss': total_loss / n_samples}
                task_log.update({
                    task_name + "_" + met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
                })
                test_log.update(task_log)
        self.logger.info(test_log)

        return test_log

class MultiaugmentSimCLRTrainer(SimCLRTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir,
            multitask_train_dataloader):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        
        self.multitask_train_dataloader = multitask_train_dataloader

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_step = 0
        for batch_idx, batch in enumerate(self.multitask_train_dataloader):
            task_name = batch['task_name']
            data, target, index  = batch['data']
            data_1, data_2 = data
            data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

            self.optimizer.zero_grad()
            h_i, h_j, z_i, z_j = self.model(data_1, data_2)
            output = h_i

            loss = self.criterion(z_i, z_j)
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

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

class MultitaskSupervisedTrainer(Trainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir,
            multitask_train_dataloader, train_data_loaders, valid_data_loaders, test_data_loaders):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        
        self.multitask_train_dataloader = multitask_train_dataloader
        self.train_data_loaders = train_data_loaders
        self.valid_data_loaders = valid_data_loaders
        self.test_data_loaders = test_data_loaders
        self.evaluator = None
        self._save_checkpoint(0)

        metric_names = ['loss', *[m.__name__ for m in self.metric_ftns]]
        metric_names = [f'{task_name}_{metric_name}' for task_name in self.valid_data_loaders.keys() for metric_name in metric_names]
        self.valid_metrics = MetricTracker(*metric_names)
        # print(self.train_metrics._data.index)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.multitask_train_dataloader):
            task_name = batch['task_name']
            data, target, index  = batch['data']
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(task_name, data)
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
        avg_loss = 0; batch_counts = 0; avg_accuracy = 0; avg_f1 = 0
        with torch.no_grad():
            for task_name, data_loader in self.valid_data_loaders.items():
                for batch_idx, (data, target, index) in enumerate(data_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(task_name, data)
                    loss = self.criterion(output, target)

                    avg_loss += loss.item()
                    batch_counts += 1

                    self.valid_metrics.update(f'{task_name}_loss', loss.item())
                    for met in self.metric_ftns:
                        res = met(output, target)
                        self.valid_metrics.update(task_name + "_" + met.__name__, res)
                        if  met.__name__ == "accuracy":
                            avg_accuracy += res
                        elif met.__name__ == "macro_f1":
                            avg_f1 += res

        valid_log = self.valid_metrics.result()
        valid_log['loss'] = avg_loss / batch_counts
        valid_log['accuracy'] = avg_accuracy / batch_counts
        valid_log['macro_f1'] = avg_f1 / batch_counts
        return valid_log
    
    def test(self):
        best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
        # if os.path.exists(best_path):
        #     first_key = torch.load(best_path, map_location=self.device)["state_dict"].keys().__iter__().__next__()
        #     if (hasattr(self.model, 'module') and first_key.startswith('module')):
        #         self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
        #     elif not hasattr(self.model, 'module'):
        #         new_state_dict = {k.replace('module.', ''): v for k, v in torch.load(best_path, map_location=self.device)["state_dict"].items()}
        #         self.model.load_state_dict(new_state_dict)
        #     else:
        #         self.model.module.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
        self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        test_log = {}
        with torch.no_grad():
            for task_name, data_loader in self.test_data_loaders.items():
                total_loss = 0.0
                total_metrics = torch.zeros(len(self.metric_ftns))
                print(len(data_loader))
                for i, (data, target, index) in enumerate(data_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(task_name, data)

                    # computing loss, metrics on test set
                    loss = self.criterion(output, target)
                    batch_size = data.shape[0]
                    total_loss += loss.item() * batch_size
                    for i, metric in enumerate(self.metric_ftns):
                        total_metrics[i] += metric(output, target) * batch_size
                n_samples = len(data_loader.sampler)
                task_log = {f'{task_name}_loss': total_loss / n_samples}
                task_log.update({
                    task_name + "_" + met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
                })
                test_log.update(task_log)
        self.logger.info(test_log)

        return test_log


from utils.tawt import get_task_weights_gradients_multi

class BilevelMultitaskTrainer(MultitaskSupervisedTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
                 train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir, 
                 multitask_train_dataloader, train_data_loaders, valid_data_loaders, test_data_loaders,
                 weight_lr=1, collect_gradient_step=1, update_weight_step = 50):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
                         train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir, 
                         multitask_train_dataloader, train_data_loaders, valid_data_loaders, test_data_loaders)
        
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
            if type(data) is list:
                data = data[0]
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(task_name, data)
            loss = self.criterion(output, target)
            loss = loss*self.task_weights[self.task_to_index[task_name]]*self.num_tasks
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

class GroupDROMultitaskTrainer(MultitaskSupervisedTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
                 train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir, 
                 multitask_train_dataloader, train_data_loaders, valid_data_loaders, test_data_loaders,
                 weight_lr=0.01, update_weight_step = 50):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
                         train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir, 
                         multitask_train_dataloader, train_data_loaders, valid_data_loaders, test_data_loaders)
        
        self.multitask_train_dataloader = multitask_train_dataloader
        self.train_data_loaders = train_data_loaders

        self.task_list = list(train_data_loaders.keys())
        self.num_tasks = len(self.task_list)
        self.task_to_index = dict([(task, i) for i, task in enumerate(self.task_list)])

        self.weight_lr = weight_lr
        self.task_weights = torch.ones((self.num_tasks, ), device=self.device)/self.num_tasks
        self.update_weight_step = update_weight_step

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_step = 1; task_loss = torch.zeros(self.num_tasks).to(self.device)
        for batch_idx, batch in enumerate(self.multitask_train_dataloader):
            task_name = batch['task_name']
            data, target, index  = batch['data']
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(task_name, data)
            loss = self.criterion(output, target)

            task_idx = self.task_to_index[task_name]
            task_loss[task_idx] += loss.detach()
            loss = loss*self.task_weights[task_idx]*self.num_tasks
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
