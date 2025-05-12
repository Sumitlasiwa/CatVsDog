import torch
import torch.nn as nn

class AlexNet(nn.Module):       # 5 conv layer + 3 FCN
    def __init__(self, num_classes=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size= 11, stride= 4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size= 5, stride= 1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(in_channels=256,out_channels= 384, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384,out_channels= 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384,out_channels= 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size= 3, stride= 2),
            nn.Flatten()    
        )
        
        self.fcn = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)    # 2 because we only need to classify cat or dog

        )
        
    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = self.fcn(feature_map)
        return output