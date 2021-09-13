# Source : https://github.com/zhang-can/PAN-PyTorch/blob/master/ops/PAN_modules.py

import torch
from torch import nn
import math

class PA(nn.Module):
    def __init__(self, n_length):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(3,8,7,1,3)

        #n_length shows the number of images which will be given as an input
        self.n_length = n_length
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        h, w = x.size(-2), x.size(-1)
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1))
        for i in range(self.n_length-1):
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1)
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        PA = d.view(-1, 1*(self.n_length-1), h, w)
        return PA