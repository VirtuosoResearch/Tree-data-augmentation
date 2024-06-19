import os
import torch
import numpy as np
import transforms
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score, log_loss

class LinearEvaluation:

    def __init__(self, model, train_dataset, valid_dataset, test_dataset, device, state_dict_dir, state_dict_name,
                task_name = None, criterion = "multiclass", num_classes = None):
        self.model = model
        self.device = device
        self.state_dict_dir = state_dict_dir
        self.state_dict_name = state_dict_name
        self.task_name = task_name
        self.criterion = criterion
        self.num_classes = num_classes
        
        train_dataset = train_dataset.copy()
        train_dataset.transform = transforms.Identity()
        valid_dataset = valid_dataset.copy()
        valid_dataset.transform = transforms.Identity()
        test_dataset = test_dataset.copy()
        test_dataset.transform = transforms.Identity()

        self.train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
        self.test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
    def eval(self):
        state_dict = torch.load(
            os.path.join(self.state_dict_dir, self.state_dict_name+".pth"), 
            map_location=self.device
        )
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        train_features = []; valid_features = []; test_features = []
        train_targets = []; valid_targets = []; test_targets = []
        with torch.no_grad():
            for data in self.train_data_loader:
                data = data.to(self.device)
                feature = self.model(data, return_features=True) if self.task_name is None else self.model(self.task_name, data, return_features=True)
                target = data.y.view(-1) if self.criterion == "multiclass" else data.y.view((-1, self.num_classes))
                train_features.append(feature.cpu().numpy())
                train_targets.append(target.cpu().numpy())

            for data in self.valid_data_loader:
                data = data.to(self.device)
                feature = self.model(data, return_features=True) if self.task_name is None else self.model(self.task_name, data, return_features=True)
                target = data.y.view(-1) if self.criterion == "multiclass" else data.y.view((-1, self.num_classes))
                valid_features.append(feature.cpu().numpy())
                valid_targets.append(target.cpu().numpy())

            for data in self.test_data_loader:
                data = data.to(self.device)
                feature = self.model(data, return_features=True) if self.task_name is None else self.model(self.task_name, data, return_features=True)
                target = data.y.view(-1) if self.criterion == "multiclass" else data.y.view((-1, self.num_classes))
                test_features.append(feature.cpu().numpy())
                test_targets.append(target.cpu().numpy())

        train_features = np.concatenate(train_features, axis=0)
        valid_features = np.concatenate(valid_features, axis=0)
        test_features  = np.concatenate(test_features, axis=0)
        train_targets  = np.concatenate(train_targets)
        valid_targets  = np.concatenate(valid_targets)
        test_targets   = np.concatenate(test_targets)

        if self.criterion == "multiclass":
            clf = svm.SVC().fit(train_features, train_targets) # LogisticRegression(max_iter=1000)

            valid_score = clf.score(valid_features, valid_targets)
            test_score = clf.score(test_features, test_targets)

            return {"eval_valid_accuracy": valid_score, "eval_test_accuracy": test_score}
        else:
            train_targets  = ((train_targets+1)/2).astype(np.long)
            valid_targets  = ((valid_targets+1)/2).astype(np.long)
            test_targets   = ((test_targets+1)/2).astype(np.long)

            clf = MultiOutputClassifier(LogisticRegression()).fit(train_features, train_targets)

            valid_prob = clf.predict_proba(valid_features)
            valid_output = clf.predict(valid_features)
            valid_loss = 0; count = 0
            for i, class_valid_prob in enumerate(valid_prob):
                if np.sum(valid_targets[:,i] == 1) > 0 and np.sum(valid_targets[:,i] == -1) > 0:
                    valid_loss += log_loss(valid_targets[:, i], class_valid_prob)
                    count += 1
            valid_loss /= count
            valid_acc = clf.score(valid_features, valid_targets)
            if np.logical_and(
                np.sum(valid_targets == 1, axis=0) > 0, 
                np.sum(valid_targets == 0, axis=0) > 0
                ).sum() == self.num_classes:
                valid_roc_auc = roc_auc_score(valid_targets, valid_output, average="macro")
            else:
                roc_list = []
                for i in range(valid_targets.shape[1]):
                    if np.sum(valid_targets[:,i] == 1) > 0 and np.sum(valid_targets[:,i] == -1) > 0:
                        roc_list.append(roc_auc_score((valid_targets, valid_output[:,i])))
                valid_roc_auc = sum(roc_list)/len(roc_list) if len(roc_list) > 0 else 0
            
            test_prob = clf.predict_proba(test_features)
            test_output = clf.predict(test_features)
            test_loss = 0
            for i, class_test_prob in enumerate(test_prob):
                test_loss += log_loss(test_targets[:, i], class_test_prob)
            test_loss /= len(test_prob)
            test_acc = clf.score(test_features, test_targets)
            if np.logical_and(
                np.sum(test_targets == 1, axis=0) > 0, 
                np.sum(test_targets == 0, axis=0) > 0
                ).sum() == self.num_classes:
                test_roc_auc = roc_auc_score(test_targets, test_output, average="macro")
            else:
                roc_list = []
                for i in range(test_targets.shape[1]):
                    if np.sum(test_targets[:,i] == 1) > 0 and np.sum(test_targets[:,i] == -1) > 0:
                        roc_list.append(roc_auc_score((test_targets, test_output[:,i])))
                
                test_roc_auc = sum(roc_list)/len(roc_list) if len(roc_list) > 0 else 0

            return {"eval_valid_loss": valid_loss/100, "eval_test_loss": test_loss/100,
                    "eval_valid_accuracy": valid_acc, "eval_test_accuracy": test_acc,
                    "eval_valid_roc_auc": valid_roc_auc, "eval_test_roc_auc": test_roc_auc}