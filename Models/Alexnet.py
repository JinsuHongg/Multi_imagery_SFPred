import torch
# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn 

class Alexnet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(Alexnet, self).__init__()
        
        # load pretrained Alexnet from github
        # convolution layers
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.features = model.features

        # fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x