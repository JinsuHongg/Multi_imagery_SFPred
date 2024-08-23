import torch
# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn 
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self) -> None:
        super(ResNet50, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained = True)
        self.model.fc = nn.Linear(2048, 2) # for binary classification
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x