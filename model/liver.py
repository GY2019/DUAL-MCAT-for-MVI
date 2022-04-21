#-*- coding:utf-8 -*-
from __future__ import division 
from __future__ import absolute_import 
from __future__ import with_statement

import os
import time
import argparse
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

from model import resnet
from model.trainer import fit
from model.metrics import AccumulatedAccuracyMetric
from utils.utils import extract_embeddings, plot_embeddings

# Device configuration
cuda = torch.cuda.is_available()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                     

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='./data/cifar10/',
                    help="""image dir path default: './data/cifar10/'.""")
parser.add_argument('--epochs', type=int, default=50,
                    help="""Epoch default:50.""")
parser.add_argument('--batch_size', type=int, default=256,
                    help="""Batch_size default:256.""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='./model/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='cifar10.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=5)

# parser.add_argument('-c', '--config', default='configs/transfer_config.json')
# # classes=('I II','III IV')
# classes=('I', 'II', 'III', 'IV')

parser.add_argument('-c', '--config', default='configs/who_config.json')
classes=('0','1','2','3','4','5')


args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)



# ##############################################################################
# #######################################Method01: Train ResNet ################
# transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, 4),
    # # transforms.RandomHorizontalFlip(p=0.50),  # 有0.75的几率随机旋转
    # # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
    # transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
# ])

# classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
# # Load data
# train_dataset = torchvision.datasets.CIFAR10(root=args.path,
                                              # transform=transform,
                                              # download=True,
                                              # train=True)

# test_dataset = torchvision.datasets.CIFAR10(root=args.path,
                                             # transform=transform,
                                             # download=True,
                                             # train=False)
# # Set up data loaders
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # 训练模型
# res_model = resnet.resnet20(num_features = 2, num_classes = 10)
# print(res_model)

# res_model.cuda()
# loss_fn = nn.NLLLoss().cuda()
# lr = 1e-2
# optimizer = torch.optim.SGD(res_model.parameters(), lr, momentum=0.9, weight_decay=5e-4)   
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
# n_epochs = 200
# log_interval = 100                           
# fit(train_loader, test_loader, res_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes=classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes=classes, save_tag = 'test')

# # Save model
# torch.save(res_model.state_dict(), './model/model_dict.pkl')

# #载入预训练Resnet
# pretrain_dict = torch.load('model/model_dict.pkl')
# res_model = resnet.resnet20(num_features = 2, num_classes = 10)
# res_model.load_state_dict(pretrain_dict)

# # 验证预训练模型
# from utils.eval import validate
# validate(test_loader, res_model.cuda(), nn.CrossEntropyLoss().cuda())
# validate(train_loader, res_model.cuda(), nn.CrossEntropyLoss().cuda())


# ##################################################################################
# #######################################Method02: ResNet for Liver ################
# ## python liver.py -c configs/transfer_config.json
# from data_loader.mri_t2wi import MRIT2WI
# from data_loader.datasets import SiameseMRI, TripletMRI
# from utils.config import get_args, process_config
# from utils.utils import printData
# config = process_config(args.config)

# # transform = transforms.Compose([
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomCrop(32, 4),
#     # transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
# # ])
# # Load data
# print('Create the data generator.')
# train_dataset = MRIT2WI(config, train = True)   
# test_dataset = MRIT2WI(config, train = False)

# # printData(test_dataset, type='normal')

# # Set up data loaders
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# res_model = resnet.resnet20(num_features = 2, num_classes = config.classes)
# print(res_model)

# res_model.cuda()
# loss_fn = nn.NLLLoss().cuda()

# lr = 1e-3
# optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# # lr = 1e-2
# # optimizer = torch.optim.SGD(res_model.parameters(), lr, momentum=0.8, weight_decay=5e-4)   
# # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
# n_epochs = 100
# log_interval = 100                           
# fit(train_loader, test_loader, res_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

# # 绘图
# linearWeights = res_model.state_dict()['linear.weight'].cpu().numpy()
# linearBias = res_model.state_dict()['linear.bias'].cpu().numpy()

# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, linearWeights, linearBias, classes=classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, linearWeights, linearBias, classes=classes, save_tag = 'test')



# ##################################################################################
# #######################################Method03: ResNet-Siamese for Liver ################
# ## python liver.py -c configs/transfer_config.json
# from data_loader.mri_t2wi import MRIT2WI
# from data_loader.datasets import SiameseMRI, TripletMRI
# from utils.config import get_args, process_config
# from utils.utils import printData
# config = process_config(args.config)

# # Load data
# print('Create the data generator.')
# train_dataset = MRIT2WI(config, train = True)   
# test_dataset = MRIT2WI(config, train = False)

# siamese_train_dataset = SiameseMRI(train_dataset) # Returns pairs of images and target same/different
# siamese_test_dataset = SiameseMRI(test_dataset)

# # printData(test_dataset, type='normal')

# # Set up data loaders
# batch_size = 32
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# from model.cifar_networks import SiameseNet
# from model.losses import ContrastiveLoss
# res_model = resnet.resnet20(3,num_features = 8, num_classes = config.classes)
# print(res_model)
# model = SiameseNet(res_model)

# model.cuda()
# loss_fn = ContrastiveLoss(1.0).cuda()

# lr = 1e-4
# optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# # lr = 1e-2
# # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)   
# # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

# n_epochs = 1000
# log_interval = 100                           
# fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

# # 绘图
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes=classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes=classes, save_tag = 'test')

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# loss_fn = nn.NLLLoss().cuda()
# model1=res_model
# optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)  
# #optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
# fit(train_loader, test_loader, model1, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
# ##################################################################################
# #######################################Method04: ResNet-Triplet for Liver ################
# ## python liver.py -c configs/transfer_config.json
# from data_loader.mri_t2wi import MRIT2WI
# from data_loader.datasets import SiameseMRI, TripletMRI
# from utils.config import get_args, process_config
# from utils.utils import printData
# config = process_config(args.config)

# # Load data
# print('Create the data generator.')
# train_dataset = MRIT2WI(config, train = True)   
# test_dataset = MRIT2WI(config, train = False)

# triplet_train_dataset = TripletMRI(train_dataset) # Returns pairs of images and target same/different
# triplet_test_dataset = TripletMRI(test_dataset)

# # printData(test_dataset, type='normal')

# # Set up data loaders
# batch_size = 32
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# from model.cifar_networks import TripletNet,classfication1Net
# from model.losses import TripletLoss
# res_model = resnet.resnet20(3,num_features = 8, num_classes = config.classes)
# print(res_model)
# model = TripletNet(res_model)

# model.cuda()
# loss_fn = TripletLoss(1.0).cuda()

# lr = 1e-4
# optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# # lr = 1e-2
# # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)   
# # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

# n_epochs = 1000
# log_interval = 100                           
# fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

# # # 绘图
# # # Set up data loaders
# # batch_size = 256
# # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# # train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# # plot_embeddings(train_embeddings_cl, train_labels_cl, classes=classes, save_tag = 'train')
# # val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# # plot_embeddings(val_embeddings_cl, val_labels_cl, classes=classes, save_tag = 'test')

# # torch.save(res_model, './model/model_dict_ss.pth')
# # print("done")
# # model1 = torch.load('./model/model_dict_ss.pth')
# # for p in model1.parameters():
# #     p.requires_grad = False



# # for p in res_model.parameters():#添加分类器并进行微调 
# #      p.requires_grad = False
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# loss_fn = nn.NLLLoss().cuda()
# model1=res_model
# optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)  
# #optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
# fit(train_loader, test_loader, model1, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])




# # from utils.eval import validate
# # validate(test_loader, model1.cuda(), nn.CrossEntropyLoss().cuda())
# # validate(train_loader, model1.cuda(), nn.CrossEntropyLoss().cuda())



#################################################################################
######################################Method05: Pretrained ResNet for Liver ################
## python liver.py -c configs/transfer_config.json
from data_loader.mri_t2wi import MRIT2WI
from data_loader.datasets import SiameseMRI, TripletMRI
from utils.config import get_args, process_config
from utils.utils import printData
config = process_config(args.config)

# transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, 4),
    # transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
# ])
# Load data
print('Create the data generator.')
train_dataset = MRIT2WI(config, train = True)   
test_dataset = MRIT2WI(config, train = False)

# printData(test_dataset, type='normal')

# Set up data loaders
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters

#载入预训练Resnet
pretrain_dict = torch.load('model/model_dict_ss.pkl')
res_model = resnet.resnet20(4,num_features = 2, num_classes = 10)
res_model.load_state_dict(pretrain_dict)

#提取fc层中固定的参数
features_num = res_model.linear.in_features
#修改类别
res_model.linear = nn.Linear(features_num, config.classes)

print(res_model)

res_model.cuda()
loss_fn = nn.NLLLoss().cuda()

lr = 1e-3
optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# lr = 1e-2
# optimizer = torch.optim.SGD(res_model.parameters(), lr, momentum=0.8, weight_decay=5e-4)   
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

n_epochs = 1000
log_interval = 100                           
fit(train_loader, test_loader, res_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

# 绘图
# Set up data loaders
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
plot_embeddings(train_embeddings_cl, train_labels_cl, classes=classes, save_tag = 'train')
val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
plot_embeddings(val_embeddings_cl, val_labels_cl, classes=classes, save_tag = 'test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
loss_fn = nn.NLLLoss().cuda()
model1=res_model
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)  
#optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
fit(train_loader, test_loader, model1, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
##################################################################################

# # L-Softmax Failed
# # ##################################################################################
# # #######################################Method06: ResNet-L-Softmax for Liver ################
# # ## python liver.py -c configs/transfer_config.json
# # from data_loader.mri_t2wi import MRIT2WI
# # from data_loader.datasets import SiameseMRI, TripletMRI
# # from utils.config import get_args, process_config
# # from utils.utils import printData
# # config = process_config(args.config)

# # # transform = transforms.Compose([
    # # # transforms.RandomHorizontalFlip(),
    # # # transforms.RandomCrop(32, 4),
    # # # transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    # # # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
# # # ])
# # # Load data
# # print('Create the data generator.')
# # train_dataset = MRIT2WI(config, train = True)   
# # test_dataset = MRIT2WI(config, train = False)

# # # printData(test_dataset, type='normal')

# # # Set up data loaders
# # batch_size = 128
# # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # # Set up the network and training parameters
# # from model import resnetA
# # res_model = resnetA.resnet20(num_features = 2, num_classes = config.classes)
# # print(res_model)

# # res_model.cuda()
# # loss_fn = nn.NLLLoss().cuda()

# # lr = 1e-3
# # optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# # # lr = 1e-2
# # # optimizer = torch.optim.SGD(res_model.parameters(), lr, momentum=0.8, weight_decay=5e-4)   
# # # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
# # n_epochs = 100
# # log_interval = 100   
# # from model.trainerA import fitA
# # fitA(train_loader, test_loader, res_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

# # # 绘图
# # linearWeights = res_model.state_dict()['linear.weight'].cpu().numpy()
# # linearBias = res_model.state_dict()['linear.bias'].cpu().numpy()

# # # Set up data loaders
# # batch_size = 256
# # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# # train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
# # plot_embeddings(train_embeddings_cl, train_labels_cl, linearWeights, linearBias, classes=classes, save_tag = 'train')
# # val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
# # plot_embeddings(val_embeddings_cl, val_labels_cl, linearWeights, linearBias, classes=classes, save_tag = 'test')


# # ##############################################################################################
# # #######################################Method06: ResNet-L-Softmax for cifar10 ################
# # transform = transforms.Compose([
    # # transforms.RandomHorizontalFlip(),
    # # transforms.RandomCrop(32, 4),
    # # # transforms.RandomHorizontalFlip(p=0.50),  # 有0.75的几率随机旋转
    # # # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
    # # transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    # # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
# # ])

# # classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
# # # Load data
# # train_dataset = torchvision.datasets.CIFAR10(root=args.path,
                                              # # transform=transform,
                                              # # download=True,
                                              # # train=True)

# # test_dataset = torchvision.datasets.CIFAR10(root=args.path,
                                             # # transform=transform,
                                             # # download=True,
                                             # # train=False)
# # # Set up data loaders
# # batch_size = 128
# # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # # Set up the network and training parameters
# # from model import resnetA
# # res_model = resnetA.resnet20(num_features = 2, num_classes = 10)
# # print(res_model)

# # res_model.cuda()
# # loss_fn = nn.NLLLoss().cuda()
# # lr = 1e-3
# # optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# # n_epochs = 100
# # log_interval = 100   
# # from model.trainerA import fitA
# # fitA(train_loader, test_loader, res_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

# # # 绘图
# # linearWeights = res_model.state_dict()['linear.weight'].cpu().numpy()
# # linearBias = res_model.state_dict()['linear.bias'].cpu().numpy()

# # # Set up data loaders
# # batch_size = 256
# # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# # train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
# # plot_embeddings(train_embeddings_cl, train_labels_cl, linearWeights, linearBias, classes=classes, save_tag = 'train')
# # val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
# # plot_embeddings(val_embeddings_cl, val_labels_cl, linearWeights, linearBias, classes=classes, save_tag = 'test')

# # # Save model
# # torch.save(res_model.state_dict(), './model/model_dict.pkl')

# # #载入预训练Resnet
# # pretrain_dict = torch.load('model/model_dict.pkl')
# # res_model = resnet.resnet20(num_features = 2, num_classes = 10)
# # res_model.load_state_dict(pretrain_dict)

# # # 验证预训练模型
# # from utils.eval import validate
# # validate(test_loader, res_model.cuda(), nn.CrossEntropyLoss().cuda())
# # validate(train_loader, res_model.cuda(), nn.CrossEntropyLoss().cuda())