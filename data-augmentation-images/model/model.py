import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vision_models
import numpy as np
from model.wideresnet import WideResNet
from model.modeling_vit import VisionTransformer, CONFIGS
import torch
import dill

class MnistModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class IdentityModule(nn.Module):
    r"""An identity module that outputs the input."""

    def __init__(self) -> None:
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x


class ResNet50(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10, **kwargs):
        super(ResNet50, self).__init__()
        self.feature_extractor = vision_models.resnet50(pretrained = pretrained)

        self.in_features = self.feature_extractor.fc.in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.fc = IdentityModule()
    
    def load_robust_model(self, state_dict_dir):
        checkpoint = torch.load(state_dict_dir, pickle_module=dill)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint["state_dict"]
        state_dict = {k[len('module.model.'):]:v for k,v in state_dict.items()}
        self.feature_extractor.load_state_dict(state_dict, strict=False)

    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x, return_features = False):
        x = self.feature_extractor(x)

        if return_features:
            return x

        x = self.pred_head(x)
        return F.log_softmax(x, dim=1)
    

class WideResNet28_10(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10, **kwargs):
        super(WideResNet28_10, self).__init__()
        self.feature_extractor = WideResNet(28, 10, dropout_rate=0.3, num_classes=n_classes)

        self.in_features = self.feature_extractor.linear.in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.linear = IdentityModule()

    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x, softmax=True, return_features = False):
        x = self.feature_extractor(x)

        if return_features:
            return x

        x = self.pred_head(x)

        if softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder_name="resnet50", projection_dim=128, pretrained = True,
                 vit_type="ViT-B_16", img_size=224, vit_pretrained_dir="pretrained/imagenet21k_ViT-B_16.npz"):
        super(SimCLR, self).__init__()

        if encoder_name == "VisionTransformer":
            vit_config = CONFIGS[vit_type]
            self.encoder = VisionTransformer(config = vit_config, img_size = img_size, zero_head=True)
            self.encoder.load_from(np.load(vit_pretrained_dir))
            self.n_features = vit_config.hidden_size
        elif encoder_name == "resnet50":
            self.encoder = ResNet50(pretrained=pretrained)
            self.n_features = self.encoder.in_features

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i, return_features=True)
        h_j = self.encoder(x_j, return_features=True)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
    
    def reset_parameters(self, state_dict):
        self.load_state_dict(state_dict)
        for module in self.projector:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

class MultitaskSimCLR(SimCLR):

    def __init__(self, encoder_name="resnet50", projection_dim=128, 
            vit_type="ViT-B_16", img_size=224, vit_pretrained_dir="pretrained/imagenet21k_ViT-B_16.npz",
            tasks = []):
        super().__init__(encoder_name, projection_dim, vit_type, img_size, vit_pretrained_dir)

        self.projector = IdentityModule()

        self.projectors = {} 

        for i, task in enumerate(tasks):
            self.projectors[task] = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, projection_dim, bias=False),
            )
        self.task_head_list = nn.ModuleDict(self.projectors)

    def reset_parameters(self, state_dict):
        self.load_state_dict(state_dict)
        for projector in self.task_head_list:
            for module in projector:
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    def forward(self, task_name, x_i, x_j):
        h_i = self.encoder(x_i, return_features=True)
        h_j = self.encoder(x_j, return_features=True)

        z_i = self.projectors[task_name](h_i)
        z_j = self.projectors[task_name](h_j)
        return h_i, h_j, z_i, z_j
    
class MultitaskResNet(ResNet50):

    def __init__(self, pretrained=True, n_classes=10, tasks = []):
        super().__init__(pretrained, n_classes)

        self.feature_extractor = vision_models.resnet50(pretrained = pretrained)
        self.feature_extractor.fc = IdentityModule()

        self.pred_heads = {}
        for task in tasks:
            self.pred_heads[task] = nn.Linear(self.in_features, self.out_features)
            
        self.task_head_dict = nn.ModuleDict(self.pred_heads)

    def reset_parameters(self, state_dict=None):
        self.load_state_dict(state_dict)
        for key, pred_head in self.task_head_dict.items():
            if isinstance(pred_head, nn.Linear):
                pred_head.reset_parameters()

    def forward(self, task_name, x, return_features=False):
        x = self.feature_extractor(x)

        if return_features:
            return x

        x = self.pred_heads[task_name](x)
        return F.log_softmax(x, dim=1)