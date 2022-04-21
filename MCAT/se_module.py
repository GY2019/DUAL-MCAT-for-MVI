from torch import nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, modality=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, modality, bias=True),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.modality = modality

    def forward(self, x):
        b, c, h, w = x.size()
        #print(x.size())
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.modality, 1, 1)#.view(b, modality, 1, 1)
        #print(y.expand(b,self.modality,h,w).size())#torch.Size([2, 8, :, :])
        #print(y.size())
        v = torch.rand((b, c, h, w)).cuda()
        for i in range(0,c):
            #print("i:",i)
            #print("c: ",int(i*self.modality//c))
            v[:,i,:,:] = y.expand(b,self.modality,h,w)[:,i*self.modality//c,:,:]
        return x * v
