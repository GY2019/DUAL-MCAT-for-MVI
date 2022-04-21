#-*- coding:utf-8 -*-
from __future__ import division 
from __future__ import absolute_import 
from __future__ import with_statement

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn import metrics
import argparse
import random
from preprogram import DataAug,resamplelabel
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.nn.functional as F
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
#from model.trainer import fit
from model.trainertwobranch import fittwo
from model.metrics import AccumulatedAccuracyMetric
from utils.utils import extract_embeddings, plot_embeddings
from MCAT.se_resnet import se_resnet20

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def drawroc(eachmodel,data_loader,criterion,tprs,aucs,mean_fpr,tpr_list,fpr_list,bestAUC,flag=1):
    criterion = nn.CrossEntropyLoss().cuda()
    model = eachmodel.cuda()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    labels=[]
    pred=[]
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    for i, (dataCT, dataMR,target)in enumerate(data_loader):#train_loader
        target1=target.float()
        print(target1)#12.3
        for i in range(0,len(target1)):
            if target1.numpy()[i] == 1:
                labels.append([0,1])
            elif target1.numpy()[i] == 0:
                labels.append([1,0])
        target = target.cuda(async=True)
        #input_var = torch.autograd.Variable(data, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)
    
        if not type(dataCT) in (tuple, list):
            data = (dataCT.cuda(),dataMR.cuda())
        if cuda:
            data = tuple(d.cuda() for d in data)
        data = tuple(Variable(d) for d in data)

        output , CTout , MRout = model(*data)
        output = F.softmax(output,dim=-1)
        loss = criterion(output, target_var)
        output1 = (output.cpu(),)
        print(output1[0].data)# to tensor type
        for i in range(0,len(output1[0].data)):
        #print(output[0].data[i])#value [torch.FloatTensor of size 1]
            pred.append(output1[0].data.numpy().tolist()[i])
    
        output = output.float()
        loss = loss.float()
    
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        """
        if i % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
        """
    print(labels)
    print(pred)
    #print(labels,pred)
    
    flabels = []
    fpred = []
    for i in range(0,len(labels)):
       flabels.append(labels[i][0])
       fpred.append(float(format(pred[i][0],'.2f')))
    
    y_label = np.array(flabels)
    y_score = np.array(fpred)
    
    print(y_label)
    print(y_score)
    
    fpr,tpr,threshold = metrics.roc_curve(y_label, y_score,pos_label=1) ###计算真正率和假正率
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc = metrics.auc(fpr,tpr) ###计算auc的值
    aucs.append(roc_auc)
    print("AUC: {:.2f} ".format(roc_auc))

    if roc_auc > bestAUC[0] :
      bestAUC[0] = roc_auc
      print("saveing the model……")
      torch.save(model.state_dict(),'./twobranchMCAT.pth')
      
    tpr_list.append(tpr)
    fpr_list.append(fpr)
    print(fpr)
    print(tpr)
    print(threshold)

def drawavgroc(tpr_list,fpr_list,aucs,mean_fpr,tprs):
    for i in range(0,len(tpr_list)):
        plt.plot(fpr_list[i], tpr_list[i],lw=1,alpha=0.3,label='%d fold ROC curve (area = %0.2f)' % (i+1,aucs[i])) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    mean_auc=metrics.auc(mean_fpr,mean_tpr)#计算平均AUC值
    std_auc=np.std(tprs,axis=0)
    std_value = np.std(aucs,ddof=1)
    plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean(AUC=%0.2f std=%0.2f)'%(mean_auc,std_value),lw=2,alpha=.8)
    print('Mean(AUC={:.2f} std={:.2f})'.format(mean_auc,std_value))
    std_tpr=np.std(tprs,axis=0)
    tprs_upper=np.minimum(mean_tpr+std_tpr,1)
    tprs_lower=np.maximum(mean_tpr-std_tpr,0)
    plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    #plt.show()
    plt.savefig("./rocresult/wholeCTMRROC.png")

parser.add_argument('-c', '--config', default='configs/who_config.json')
classes=('0','1')


args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)


## python liver.py -c configs/transfer_config.json
from data_loader.ZHEloader import MRIORCT,MRIANDCT
from data_loader.datasets import SiameseMRI, TripletMRI
from utils.config import get_args, process_config
from utils.utils import printData
config = process_config(args.config)
# Load data
print('Create the data generator.')

wholedata_path=[]
fp = open(os.path.join(config.data_path, "MRICTdata.txt"), 'r')
line = fp.readline()
while len(line):
    wholedata_path.append(line)
    line = fp.readline()
fp.close()
#random.shuffle(wholedata_path)
#wholedata_path = resamplelabel(wholedata_path)
"""
triplet_train_dataset = TripletMRI(train_dataset) # Returns pairs of images and target same/different
triplet_test_dataset = TripletMRI(test_dataset)
"""
# printData(test_dataset, type='normal')

# Set up data loaders

# batch_size = 32
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from model.cifar_networks import TripletNet
from model.losses import TripletLoss
from model.losses import OnlineTripletLoss
# Strategies for selecting triplets within a minibatch
from utils.utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector 
from utils.utils import extract_embeddings, plot_embeddings

numkf=5
kf = KFold(n_splits=numkf)
ZHE_times=1
besttrainacc = 0
bestvalacc = 0
bestAUC = [0]
avgtrain=[]
avgval=[]
final=[]
avgacc=[]
avgspec=[]
avgsens=[]
avgauu=[]

Sens=[]
Prec=[]
F1=[]
cnf_mat=[]
cnf_mat2 = np.zeros((2, 2))
avgmax=[]

tprs=[]
aucs=[]
tpr_list = []
fpr_list = []
mean_fpr=np.linspace(0,1,100)
plt.figure()

for train_index, test_index in kf.split(wholedata_path):
    print("----------------------------------------")
    print("the"+str(ZHE_times)+"th ZHE experiment")

    train_select,test_select=np.array(wholedata_path)[train_index], np.array(wholedata_path)[test_index]
    print(train_select.shape)
    print(test_select.shape)

    flag=True
    batch_size = 16
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # Load data
    print('Create the data generator.')
    train_dataset = MRIANDCT(config, resamplelabel(train_select),train = True,classify = True)#MRI = False -- CT   
    #train_dataset = MRIANDCT(config, resamplelabel(DataAug(train_select)),train = True,classify = True)#MRI = False -- CT
    test_dataset = MRIANDCT(config, test_select, train = False,classify = True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    res_modelCT = se_resnet20(num_classes=64, modality=3)
    res_modelMR = se_resnet20(num_classes=64, modality=6)
    
    CT_classifer = resnet.ClassificationNet1(res_modelCT,64, config.classes)
    MR_classifer = resnet.ClassificationNet1(res_modelMR,64, config.classes)
    
    CT_classifer.load_state_dict(torch.load('./CTbranch.pth'))
    MR_classifer.load_state_dict(torch.load('./MRIbranch.pth'))
    
    twobranchmodel = resnet.twobranchClassificationNet(CT_classifer,MR_classifer)
    loss_fn = nn.NLLLoss().cuda() 
    lr1 = 1e-2#EMCANET 1e-2
    
    n_epochs = 100#100
    log_interval = 40

    #model1=res_model
    twobranchmodel.cuda()
    optimizer1 = torch.optim.SGD(twobranchmodel.parameters(), lr1, momentum=0.9, weight_decay=5e-4) #[twobranchmodel.alpha]
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.85, last_epoch=-1)#step_size=5 EMCANET 0.85 #MCAT LOSSTOTAL 0.75
    a,b,e,Sens1, Prec1, F11, cnf_mat1,k=fittwo(train_loader, test_loader, twobranchmodel, loss_fn, optimizer1, scheduler1, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()],flag=flag)
    print("ct branch weight:",twobranchmodel.alpha)
    #print(model1.embedding_net.conv1.weight)
    Sens.append(max(Sens1))
    Prec.append(max(Prec1))
    F1.append(max(F11))
    cnf_mat.append(cnf_mat1)
    for i in cnf_mat1:
        #print(i.shape)
        cnf_mat2 += i
    #print Sens, Prec, F1, cnf_mat
    avgtrain.append(a)
    avgval.append(b)
    final.append(e)
    avgmax.append(k)

    from utils.evaltwobranch import validate
    validate(test_loader, twobranchmodel.cuda(), nn.CrossEntropyLoss().cuda())
    validate(train_loader, twobranchmodel.cuda(), nn.CrossEntropyLoss().cuda())
    print("final rate: {}".format(e))
    if e > bestvalacc :
      bestvalacc = e
      besttrainacc = a
      #print("saveing the {} zhe model……".format(ZHE_times))
      #torch.save(twobranchmodel.state_dict(),'./twobranchmodelMCAT.pth')
    elif e==bestvalacc:
      if a > besttrainacc :
          besttrainacc = a
          #print("saveing the {} zhe model……".format(ZHE_times))
          #torch.save(twobranchmodel.state_dict(),'./twobranchmodelMCAT.pth')
    avgtrain1=0
    avgval1=0
    for s in avgtrain:
        avgtrain1+=int(s)
    for d in avgval:
        avgval1+=int(d)
    u=max(avgtrain)
    v=max(avgval)

    # # Set up data loaders
    # batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    ZHE_times +=1
    drawroc(twobranchmodel.cuda(),test_loader,nn.CrossEntropyLoss().cuda(),tprs,aucs,mean_fpr,tpr_list,fpr_list,bestAUC,flag=1)
drawavgroc(tpr_list,fpr_list,aucs,mean_fpr,tprs)
print(cnf_mat2)
print(final)
print(Sens)
print(Prec)
print(F1)
print("avgtrainacc for "+str(numkf)+" times "+str(avgtrain1/numkf))
print("avgtrainacc for "+str(numkf)+" times "+str(avgtrain1/numkf))
print("avgvalacc for "+str(numkf)+" times "+str(avgval1/numkf))
print("maxtrainacc for "+str(numkf)+" times "+str(u))
print("maxvalacc for "+str(numkf)+" times "+str(v))
print("--------------------------------")
print("avgmax for "+str(numkf)+" times "+str(sum(avgmax)/numkf))
avgmax1=np.array(avgmax)
print("avgmax std for "+str(numkf)+" times "+str(np.std(avgmax1)))
print("maxfinalvalacc for "+str(numkf)+" times "+str(max(final)))
print("avgfinalvalacc for "+str(numkf)+" times "+str(sum(final)/numkf))
for i in range(len(final)):
    final[i]/=100
nfinal=np.array(final)
print("var for "+str(numkf)+" times "+str(np.std(nfinal)))

print("Sens "+str(numkf)+" times "+str(sum(Sens)/len(Sens)))
print("Prec "+str(numkf)+" times "+str(sum(Prec)/len(Prec)))
print("F1 "+str(numkf)+" times "+str(sum(F1)/len(F1)))

Sens=np.array(Sens)
Prec=np.array(Prec)
F1=np.array(F1)
print("Sens std "+str(numkf)+" times "+str(np.std(Sens)))
print("Prec stf"+str(numkf)+" times "+str(np.std(Prec)))
print("F1 std"+str(numkf)+" times "+str(np.std(F1)))