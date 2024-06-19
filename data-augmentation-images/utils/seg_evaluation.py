import os
import torch
import numpy as np

from etab.baselines.models import ETABmodel
from etab.utils.metrics import evaluate_model
class SegEvaluation:

    def __init__(self, model, train_data_loader, valid_data_loader, test_data_loader, device, state_dict_dir, state_dict_name,
                 eval_lr=0.001, eval_n_epoch=10):
        self.model = model
        self.device = device
        self.state_dict_dir = state_dict_dir
        self.state_dict_name = state_dict_name
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.eval_lr = eval_lr
        self.eval_n_epoch = eval_n_epoch
        
    def eval(self):
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

        seg_model = ETABmodel(task="segmentation",
                       backbone='ResNet-50', 
                       head="SegFormer")
        
        # Assume using ResNet-50
        if hasattr(self.model, "module"):
            state_dict = self.model.module.encoder.feature_extractor.state_dict() 
        else:
            state_dict = self.model.encoder.feature_extractor.state_dict() 
        seg_model.backbone.load_state_dict(state_dict, strict=False)
        seg_model.freeze_backbone()

        train_log = seg_model.fit(train_loader=self.train_data_loader, 
                valid_loader=self.valid_data_loader, 
                task_code="a0-A4-E", 
                n_epoch=self.eval_n_epoch,
                learning_rate=self.eval_lr,
                ckpt_dir="a0-A4-E", 
                device=self.device)
        
        test_log = evaluate_model(seg_model, self.test_data_loader)


        return {
            "eval_valid_loss": train_log["val_loss"], "eval_test_loss": test_log["test_loss"],
            "eval_valid_accuracy": train_log["val_acc"], "eval_test_accuracy": test_log["test_acc"],
            "eval_valid_fscore": train_log["val_fscore_macro"], "eval_test_fscore": test_log["test_fscore_macro"],
            "eval_valid_dice": train_log["val_jaccard_index"], "eval_test_dice": test_log["test_jaccard_index"],
            }