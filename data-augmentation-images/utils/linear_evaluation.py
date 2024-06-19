import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from data_loader.data_loaders import Cifar10DataLoader
from transforms.lisa import LISAMixUp

class LinearEvaluation:

    def __init__(self, model, train_data_loader, valid_data_loader, test_data_loader, device, state_dict_dir, state_dict_name, 
                state_dict=None):
        self.model = model
        self.device = device
        self.state_dict_dir = state_dict_dir
        self.state_dict_name = state_dict_name
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.state_dict = state_dict
        self.prepare_features()
    
    def prepare_features(self):
        if self.state_dict is not None:
            state_dict = self.state_dict
        else:
            state_dict = torch.load(
                os.path.join(self.state_dict_dir, self.state_dict_name+".pth"), 
                map_location=self.device
            )['state_dict']
        new_state_dict = {}
        for key, val in state_dict.items():
            if "module" in key:
                new_state_dict[key.replace("module.", "")] = val
            else:
                new_state_dict[key] = val
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)

        train_features = []; valid_features = []; test_features = []
        train_targets = []; valid_targets = []; test_targets = []
        with torch.no_grad():
            for (data, target, _) in self.train_data_loader:
                data = data.to(self.device)
                if hasattr(self.model, "module"):
                    feature = self.model.module.encoder(data, return_features=True)
                else:
                    feature = self.model.encoder(data, return_features=True)
                train_features.append(feature.cpu().numpy())
                train_targets.append(target.cpu().numpy())

            for (data, target, _) in self.valid_data_loader:
                data = data.to(self.device)
                if hasattr(self.model, "module"):
                    feature = self.model.module.encoder(data, return_features=True)
                else:
                    feature = self.model.encoder(data, return_features=True)
                valid_features.append(feature.cpu().numpy())
                valid_targets.append(target.cpu().numpy())

            for (data, target, _) in self.test_data_loader:
                data = data.to(self.device)
                if hasattr(self.model, "module"):
                    feature = self.model.module.encoder(data, return_features=True)
                else:
                    feature = self.model.encoder(data, return_features=True)
                test_features.append(feature.cpu().numpy())
                test_targets.append(target.cpu().numpy())

            # if self.train_lisa:
            #     processed_train_features = []
            #     for i in range(len(train_features)):
            #         same_y_idx = np.where(train_targets == train_targets[i])[0]
            #         if len(same_y_idx) == 0:
            #             processed_train_features.append(train_features[i])
            #         else:
            #             mix_list = [train_features[j] for j in same_y_idx]
            #             processed_train_features.append(LISAMixUp(mix_list,0.5)(train_features[i]))
            #     train_features = processed_train_features
            
        self.train_features = np.concatenate(train_features, axis=0)
        self.valid_features = np.concatenate(valid_features, axis=0)
        self.test_features = np.concatenate(test_features, axis=0)
        self.train_targets = np.concatenate(train_targets)
        self.valid_targets = np.concatenate(valid_targets)
        self.test_targets = np.concatenate(test_targets)

        if not os.path.exists('linear_result'):
            os.makedirs('linear_result')

        np.save('linear_result/trn_f.npy',  np.concatenate(train_features, axis=0))
        np.save('linear_result/val_f.npy', np.concatenate(valid_features, axis=0))
        np.save('linear_result/tst_f.npy',  np.concatenate(test_features, axis=0))
        np.save('linear_result/trn_t.npy', np.concatenate(train_targets))
        np.save('linear_result/val_t.npy', np.concatenate(valid_targets))
        np.save('linear_result/tst_t.npy', np.concatenate(test_targets))

    def eval(self):
        clf = LogisticRegression(max_iter=1000).fit(self.train_features, self.train_targets)

        valid_probs = clf.predict_proba(self.valid_features)
        test_probs = clf.predict_proba(self.test_features)
        valid_loss = log_loss(self.valid_targets, valid_probs)
        test_loss = log_loss(self.test_targets, test_probs)
        valid_score = clf.score(self.valid_features, self.valid_targets)
        test_score = clf.score(self.test_features, self.test_targets)

        return {
            "eval_valid_loss": valid_loss, "eval_test_loss": test_loss,
            "eval_valid_accuracy": valid_score, "eval_test_accuracy": test_score
            }