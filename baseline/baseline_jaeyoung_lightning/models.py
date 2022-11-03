import timm
import torch.nn as nn
import torchvision


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = torchvision.models.efficientnet_b0(pretrained=True)
        self.fc = nn.Linear(in_features=self.base_model.classifier[1].out_features, out_features=num_classes)

    def forward(self, x):
        x=self.base_model(x)
        x=self.fc(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
        print(self.base_model)
        self.fc = nn.Linear(in_features=self.base_model.head.out_features, out_features=num_classes)

    def forward(self, x):
        x=self.base_model(x)
        x=self.fc(x)
        return x

class ResnextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = timm.create_model('resnext50_32x4d',pretrained=True,num_classes=num_classes)

    def forward(self, x):
        return self.base_model(x)