import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__ (self, in_feature: int, hidden_feature: int, hidden_layers: int, out_feature: int):
        super().__init__()

        # first layer
        layers = [
                nn.Linear(in_feature, hidden_feature),
                nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
                # nn.Dropout(p=0.5)
                
        ]

        # last layer
        out_layer_list = [
            nn.Linear(hidden_feature, out_feature),
            nn.Sigmoid()
        ]

        # hidden layers
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(hidden_feature, hidden_feature),
                nn.BatchNorm1d(hidden_feature),
                nn.ReLU()
            ])
        
        layers.extend(out_layer_list)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




