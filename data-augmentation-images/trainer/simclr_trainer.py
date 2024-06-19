from trainer.base_trainer import Trainer
import os
import torch
from utils.linear_evaluation import LinearEvaluation
from torchvision import transforms
import copy
from transforms import TestTransform

class SimCLRTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader=None, test_data_loader=None, lr_scheduler=None, checkpoint_dir=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        self.evaluator = None
        self._save_checkpoint(0)
        

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, ((data_1, data_2), target, index) in enumerate(self.train_data_loader):
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
            for batch_idx, ((data_1, data_2), target, index) in enumerate(self.valid_data_loader):
                data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

                h_i, h_j, z_i, z_j = self.model(data_1, data_2)
                loss = self.criterion(z_i, z_j)
                output = h_i

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()
    
    def test(self, load_best=True):
        best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
        if load_best and os.path.exists(best_path):
            first_key = torch.load(best_path, map_location=self.device)["state_dict"].keys().__iter__().__next__()
            if (hasattr(self.model, 'module') and first_key.startswith('module')):
                self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
            elif not hasattr(self.model, 'module'):
                new_state_dict = {k.replace('module.', ''): v for k, v in torch.load(best_path, map_location=self.device)["state_dict"].items()}
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.module.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
            print(f"Loaded the best model from {best_path}")
        self.model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))

        with torch.no_grad():
            for i, ((data_1, data_2), target, index) in enumerate(self.test_data_loader):
                data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

                h_i, h_j, z_i, z_j = self.model(data_1, data_2)
                output = h_i

                # computing loss, metrics on test set
                loss = self.criterion(z_i, z_j)
                batch_size = data_1.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_ftns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(self.test_data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })
        self.logger.info(log)

        return log
    
    def set_evaluator(self):
        if self.config['data_loader']['type'] == "MessidorDataLoader":
            supervised_train_transform = transforms.Compose([
                        transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                        transforms.Pad((0, 37, 0, 38)),
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                    ])
            supervised_test_transform = transforms.Compose([
                        transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                        transforms.Pad((0, 37, 0, 38)),
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                    ])
        elif self.config['data_loader']['type'] == "JinchiDataLoader":
            supervised_train_transform = transforms.Compose([
                        transforms.Resize((224, 224)), 
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                    ])
            supervised_test_transform = transforms.Compose([
                        transforms.Resize((224, 224)), 
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                    ])
        elif self.config['data_loader']['type'] == "AptosDataLoader":
            supervised_train_transform = transforms.Compose([
                        transforms.Resize(224), 
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                    ])
            supervised_test_transform = transforms.Compose([
                        transforms.Resize(224), 
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                    ])
        elif self.config['data_loader']['type'] == "Cifar10DataLoader" or self.config['data_loader']['type'] == "Cifar100DataLoader":
            supervised_train_transform = TestTransform()
            supervised_test_transform = TestTransform()
        else:
            print("Unknown data loader type!")
        eval_train_loader = copy.deepcopy(self.train_data_loader)
        eval_valid_loader = copy.deepcopy(self.valid_data_loader)
        eval_test_loader = copy.deepcopy(self.test_data_loader)
        eval_train_loader.dataset.transform = supervised_train_transform
        eval_valid_loader.dataset.transform = supervised_test_transform
        eval_test_loader.dataset.transform = supervised_test_transform
        self.evaluator = LinearEvaluation(self.model, eval_train_loader, eval_valid_loader, eval_test_loader,
                                    self.device, state_dict_dir=self.checkpoint_dir, state_dict_name="model_best", 
                                    state_dict=self.model.state_dict())