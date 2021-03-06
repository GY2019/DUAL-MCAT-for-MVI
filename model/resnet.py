'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel, num_features):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_features)
        
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.shape)
        # raw_input()
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
    def get_embedding(self, x):
        return self.forward(x)

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, num_features, num_classes):
        super(ClassificationNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_net = embedding_net
        self.linear = nn.Linear(num_features, num_classes, bias=False)
        
        self.apply(_weights_init)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.linear(output)
        scores = F.log_softmax(output, dim=-1)
        return scores

    def get_embedding(self, x):
        return self.embedding_net(x)
        
    def _get_name(self):
        return 'ClassificationNet'

class ClassificationNet1(nn.Module):
    def __init__(self, embedding_net, num_features, num_classes):
        super(ClassificationNet1, self).__init__()
        self.num_classes = num_classes
        self.embedding_net = embedding_net
        self.linear1 = nn.Linear(num_features, 16, bias=False)
        self.linear2 = nn.Linear(32, 10, bias=False)
        self.linear4 = nn.Linear(256, 16, bias=False)
        self.linear3 = nn.Linear(16, num_classes, bias=False)
        self.dp = nn.Dropout(0.5)
        self.apply(_weights_init)

    def forward(self, x):
        output = self.embedding_net(x)
        #output = self.dp(output)
        output = self.linear1(output)
        #output = self.dp(output)
        #output = self.linear2(output)
        #output = self.dp(output)
        #output = self.linear4(output)
        #output = self.dp(output)
        output = self.linear3(output)
        scores = F.log_softmax(output, dim=-1)
        #scores = F.softmax(output,dim=-1)
        return scores

    def get_embedding(self, x):
        return self.embedding_net(x)
        
    def _get_name(self):
        return 'ClassificationNet'

class DeepandtraditionClassificationNet(nn.Module):
    def __init__(self, embedding_net,num_features,num_classes):
        super(DeepandtraditionClassificationNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_net = embedding_net
        self.linear1 = nn.Linear(num_features, 32, bias=False)
        self.linear2 = nn.Linear(32, 16, bias=False)
        self.linear4 = nn.Linear(256, 16, bias=False)
        self.linear3 = nn.Linear(16, num_classes, bias=False)
        self.dp = nn.Dropout(0.5)
        self.apply(_weights_init)

    def forward(self, x1, x2):
        output = self.embedding_net(x1,x2)
        #output = self.dp(output)
        output = F.relu(self.linear1(output))
        #output = self.dp(output)
        #output = self.linear2(output)
        output = F.relu(self.linear2(output))
        #output = self.dp(output)
        #output = self.linear4(output)
        #output = self.dp(output)
        output = self.linear3(output)
        scores = F.log_softmax(output, dim=-1)
        #scores = F.softmax(output,dim=-1)
        return scores

    def get_embedding(self, x):
        return self.embedding_net(x)
        
    def _get_name(self):
        return 'ClassificationNet'

class twobranchClassificationNet(nn.Module):
    def __init__(self, classifer1, classifer2):
        super(twobranchClassificationNet, self).__init__()
        self.classifer1 = classifer1
        self.classifer2 = classifer2
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.5)
        #self.alpha.requires_grad = False####
        
    def forward(self, x1,x2):
        output1 = self.classifer1(x1)
        #output = self.dp(output)
        output2 = self.classifer2(x2)
        output = self.alpha*F.softmax(output1,dim=-1) + (1-self.alpha)*F.softmax(output2,dim=-1)
        #print(output)
        scores = F.log_softmax(output, dim=-1)
        #scores = F.softmax(output,dim=-1)#test draw roc used
        #print(scores)
        #scores,F.log_softmax(output1, dim=-1),F.log_softmax(output2, dim=-1)
        return scores,output1,output2
        
    def _get_name(self):
        return 'twobranchClassificationNet'

def resnet20(in_channel, num_features = 64, num_classes = 10):
    embedding_net = ResNet(BasicBlock, [3, 3, 3], in_channel, num_features)
    #resnet = ClassificationNet(embedding_net, num_features, num_classes)
    resnet=embedding_net
    return resnet


def resnet32(in_channel, num_features = 64, num_classes = 10):
    embedding_net = ResNet(BasicBlock, [5, 5, 5], in_channel, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet
    
    
def resnet44(in_channel, num_features = 64, num_classes = 10):
    embedding_net = ResNet(BasicBlock, [7, 7, 7], in_channel, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet


def resnet56(in_channel, num_features = 64, num_classes = 10):
    embedding_net = ResNet(BasicBlock, [9, 9, 9], in_channel, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet


def resnet110(in_channel, num_features = 64, num_classes = 10):
    embedding_net = ResNet(BasicBlock, [18, 18, 18], in_channel, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet


def resnet1202(in_channel, num_features = 64, num_classes = 10):
    embedding_net = ResNet(BasicBlock, [200, 200, 200], in_channel, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()