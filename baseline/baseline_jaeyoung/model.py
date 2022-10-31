import timm
import torch.nn as nn
import torchvision


class ResnextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = torchvision.models.resnext50_32x4d(pretrained=True)
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1000, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        )
    def forward(self, x):
        return self.base_model(x)

class Deit3Base224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = timm.create_model('deit3_base_patch16_224_in21ft1k',pretrained=True)
        self.base_model.head = nn.Linear(in_features=self.base_model.head.in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.base_model(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(in_features=self.base_model.fc.in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.base_model(x)


class ConvnextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = torchvision.models.convnext_base(pretrained=True)
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1000, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        )
    def forward(self, x):
        return self.base_model(x)

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = torchvision.models.efficientnet_b4(pretrained=True)
        self.fc = nn.Linear(in_features=self.base_model.classifier[1].out_features, out_features=num_classes)

    def forward(self, x):
        x=self.base_model(x)
        x=self.fc(x)
        return x

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