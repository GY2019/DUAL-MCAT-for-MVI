# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self, in_channel = 1, out_num = 2):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channel, 32, 3), nn.PReLU(),# 3*32*32 ==> 32*30*30
                                     nn.Conv2d(32, 32, 3), nn.PReLU(),        # 32*30*30==> 32*28*28
                                     nn.MaxPool2d(2, stride=2),               # 32*28*28==> 32*14*14
                                     nn.Conv2d(32, 64, 3), nn.PReLU(),        # 32*14*14==> 32*12*12
                                     nn.Conv2d(64, 64, 3), nn.PReLU(),        # 32*12*12==> 32*10*10
                                     nn.MaxPool2d(2, stride=2))               # 32*10*10==> 32*5*5
                                     
        self.fc = nn.Sequential(nn.Linear(64 * 5 * 5, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, out_num) #原始代码为了画图编码成2维，可以改成更高维
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, input_num, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(input_num, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class classfication1Net(nn.Module):
    def __init__(self, embedding_net,n_classes):
        super(classfication1Net, self).__init__()
        self.embedding_net = embedding_net
        self.L1=nn.Linear(n_classes*3,n_classes)

    def forward(self, x1, x2, x3):
        output1,output2,output3 = self.embedding_net(x1,x2,x3)
        output=torch.cat([output1,output2,output3],1)
        scores = F.log_softmax(self.L1(output), dim=1)
        return scores

    def get_embedding(self, x):
        return self.embedding_net(x)
