from trainer.base_trainer import Trainer
from utils import num_graphs, MetricTracker
import torch
from models.losses import criterions
from sklearn.metrics import roc_auc_score

class MultitaskTrainer(Trainer):

    def __init__(self, model, optimizer, lr_scheduler, criterion, metric_ftns,
                 train_loader, valid_loader, test_loader, device, logger, epochs,
                 save_epochs, checkpoint_dir, mnt_metric, mnt_mode,
                 multitask_train_dataloader, valid_loaders, test_loaders,
                 task_criterions, task_metrics, loss_weight=0.1, use_supervised_loss=True):
        super().__init__(model, optimizer, lr_scheduler, criterion, metric_ftns, 
                         train_loader, valid_loader, test_loader, device, logger, epochs, 
                         save_epochs, checkpoint_dir, mnt_metric, mnt_mode)
        self.multitask_train_dataloader = multitask_train_dataloader
        self.valid_loaders = valid_loaders
        self.test_loaders = test_loaders

        self.use_supervised_loss = use_supervised_loss
        self.lam = loss_weight
        self.task_to_criterions = {task: task_criterions[i] for i, task in enumerate(self.valid_loaders.keys())}
        self.task_to_metrics = {task: task_metrics[i] for i, task in enumerate(self.valid_loaders.keys())}
        self.task_to_train_metrics = {task: MetricTracker('loss', *[m.__name__ for m in self.task_to_metrics[task]]) for task in self.valid_loaders.keys()}
        self.task_to_valid_metrics = {task: MetricTracker('loss', *[m.__name__ for m in self.task_to_metrics[task]]) for task in self.valid_loaders.keys()}
        self.task_to_test_metrics  = {task: MetricTracker('loss', *[m.__name__ for m in self.task_to_metrics[task]]) for task in self.test_loaders.keys()}

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        for metric in self.task_to_train_metrics.values():
            metric.reset()
        for batch in self.multitask_train_dataloader:
            task_name = batch['task_name'][0]
            data, data1, data2 = batch['sample']
            if num_graphs(data1) == 1:
                continue
            self.optimizer.zero_grad()
            
            data = data.to(self.device)
            data1 = data1.to(self.device)
            data2 = data2.to(self.device)

            # self-supervised loss
            out1 = self.model.forward_cl(task_name, data1)
            out2 = self.model.forward_cl(task_name, data2)
            unsupervised_loss = criterions["info_nce"](out1, out2)

            # supervised loss
            if self.use_supervised_loss:
                task_criterion = self.task_to_criterions[task_name]
                output = self.model(task_name, data, return_softmax=(task_criterion == "multiclass"))
                target = (data.y.view(output.shape).to(torch.float64)+1)/2 if task_criterion == "multilabel" else data.y.view(-1)
                supervised_loss = criterions[task_criterion](output, target)
            else:
                task_criterion = self.task_to_criterions[task_name]
                output = out1
                target = (data.y.view(output.shape).to(torch.float64)+1)/2 if task_criterion == "multilabel" else data.y.view(-1)
                supervised_loss = 0

            loss = supervised_loss + self.lam*unsupervised_loss
            loss.backward()
            self.optimizer.step()
            
            task_metric = self.task_to_train_metrics[task_name]
            task_metric.update('loss', loss.item())
            for met in self.task_to_metrics[task_name]:
                output = output
                target = target
                task_metric.update(met.__name__, met(output, target).item(), n=num_graphs(data1))

        log = {}; avg_loss = 0
        for task_name in self.task_to_train_metrics.keys():
            task_log = self.task_to_train_metrics[task_name].result()
            avg_loss += task_log['loss']
            log.update(**{f"{task_name}_{key}": val for key, val in task_log.items()})
        log.update(**{'loss': avg_loss/len(self.task_to_train_metrics.keys())})

        if self.valid_loader is not None:
            val_log = self.eval(phase="valid")
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.valid_loader is not None:
            val_log = self.eval(phase="test")
            log.update(**{'test_'+k : v for k, v in val_log.items()})

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
            y_true = []
            y_scores = []

            with torch.no_grad():
                for batch_idx, (data, data1, data2) in enumerate(data_loader):
                    data = data.to(self.device)
                    data1 = data1.to(self.device)
                    data2 = data2.to(self.device)

                    out1 = self.model.forward_cl(task_name, data1)
                    out2 = self.model.forward_cl(task_name, data2)
                    unsupervised_loss = self.criterion(out1, out2)
                    
                    # supervised loss
                    if self.use_supervised_loss:
                        task_criterion = self.task_to_criterions[task_name]
                        output = self.model(task_name, data, return_softmax=(task_criterion == "multiclass"))
                        target = (data.y.view(output.shape).to(torch.float64)+1)/2 if task_criterion == "multilabel" else data.y.view(-1)
                        supervised_loss = criterions[task_criterion](output, target)
                    else:
                        task_criterion = self.task_to_criterions[task_name]
                        output = out1
                        target = (data.y.view(output.shape).to(torch.float64)+1)/2 if task_criterion == "multilabel" else data.y.view(-1)
                        supervised_loss = 0
                    
                    loss = supervised_loss + self.lam*unsupervised_loss

                    metrics.update('loss', loss.item())
                    
                    y_true.append(target)
                    y_scores.append(output)
                
            y_true = torch.cat(y_true, dim = 0)
            y_scores = torch.cat(y_scores, dim = 0)

            for met in self.task_to_metrics[task_name]: 
                metrics.update(met.__name__, met(y_scores, y_true).item(), n=1)
                # if met.__name__ == "roc_auc":
                #     roc_list = []
                #     for i in range(y_true.shape[1]):
                #         if torch.sum(y_true[:,i] == 1) > 0 and torch.sum(y_true[:,i] == 0) > 0:
                #             roc_list.append(roc_auc_score((y_true[:,i]).cpu().numpy(), y_scores[:,i].cpu().numpy()))
                #     metrics.update(met.__name__, sum(roc_list)/len(roc_list), n=1)  
                # else:   
                #     metrics.update(met.__name__, met(y_scores, y_true).item(), n=1)
            
            task_log = metrics.result() 
            avg_loss += task_log['loss']
            task_log = {f"{task_name}_{key}": val for key, val in task_log.items()}
            log.update(**task_log)
        log.update(**{'loss': avg_loss/len(task_data_loaders.keys())})
        return log