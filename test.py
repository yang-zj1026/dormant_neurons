import torch
import torch.nn as nn


class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10, 20),
                                     nn.ReLU(),
                                     nn.Linear(20, 10),
                                     nn.ReLU(),
                                     )


mymodel = Mymodel()

for name, param in mymodel.named_parameters():
    print(name, param.data)
