from torch.autograd import Variable
import numpy as np
from utils.utils import modelsize
from utils1 import *   

def fittwo(train_loader, val_loader, model, loss_fn, optimizerCT,schedulerCT,optimizerMR,schedulerMR,optimizer,scheduler,n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0,flag=0,select=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        if select == 0:
            schedulerCT.step()
        elif select == 1:
            schedulerMR.step()
        elif select == 2:
            scheduler.step()
    j=0
    
    temp=0.0
    temp1=0.0
    temp3=0.0
    Sens1=[]
    Prec1=[]
    F11=[]
    cnf_mat1=[]

    avgmax=[]
    for epoch in range(start_epoch, n_epochs):
        if select == 0:
            schedulerCT.step()
        elif select == 1:
            schedulerMR.step()
        elif select == 2:
            scheduler.step()
        lr3 = scheduler.get_lr()
        lr1 = schedulerCT.get_lr()
        lr2 = schedulerMR.get_lr()
        Sens=[]
        Prec=[]
        F1=[]
        cnf_mat=[]
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizerCT,optimizerMR,optimizerCT,cuda, log_interval, metrics,epoch,flag=flag,select=select)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}. lrCT:{} lrMR:{} lrTT:{}'.format(epoch + 1, n_epochs, train_loss,lr1,lr2,lr3)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            temp1+=metric.value()
        val_loss, metrics,Sens, Prec, F1, cnf_mat = test_epoch(val_loader, model, loss_fn, cuda, metrics,epoch,flag=flag)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}. lrCT:{} lrMR:{} lrTT:{}'.format(epoch + 1, n_epochs,val_loss,lr1,lr2,lr3)
        
                                                                        
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            temp+=metric.value()
            avgmax.append(metric.value())
            j=j+1
            if epoch==n_epochs-1:
                temp3=metric.value()
        if epoch==n_epochs-1:
            Sens1, Prec1, F11, cnf_mat1=Sens, Prec, F1, cnf_mat
            #print Sens1, Prec1, F11, cnf_mat1
        print(message)
    print("avgtrain_acc="+str(temp1/n_epochs))
    print("avgval_acc="+str(temp/n_epochs))
    if flag==False:
        return temp1/n_epochs,temp/n_epochs,temp3,Sens1, Prec1, F11, cnf_mat1,Sens
    return temp1/n_epochs,temp/n_epochs,temp3,Sens1, Prec1, F11, cnf_mat1,max(avgmax)
    
def train_epoch(train_loader, model, loss_fn, optimizerCT,optimizerMR,optimizer, cuda, log_interval, metrics,epoch,flag,select):
    acc = []
    sens = []
    spec = []
    auc = []

    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
   
    for batch_idx, (dataCT, dataMR,target) in enumerate(train_loader):
        #print("____________________")
        #print(len(train_loader)) #12  = 384 /32
        lengthdata = len(dataCT)# additional
        temp=target
        target = target if len(target) > 0 else None
        if not type(dataCT) in (tuple, list):
            data = (dataCT.cuda(),dataMR.cuda())
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        data = tuple(Variable(d) for d in data)

        optimizerCT.zero_grad()
        optimizerMR.zero_grad()
        optimizer.zero_grad()
        outputs , CTout , MRout = model(*data)
        """
        _ , CTout , MRout = model(*data)
        outputs = CTout*0.5+MRout*0.5
        """
        #print("____________________")
        #print(target)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
            CTout = (CTout,)
            MRout = (MRout,)

        loss_inputs = outputs
        loss_inputct = CTout
        loss_inputmr = MRout
        if target is not None:
            target = Variable(target)
            target = (target,)
            loss_inputs += target
            loss_inputct += target
            loss_inputmr += target
        #print(loss_inputs)
        loss_outputs = loss_fn(*loss_inputs)
        loss_outputct = loss_fn(*loss_inputct)
        loss_outputmr = loss_fn(*loss_inputmr)
        
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        lossct = loss_outputct[0] if type(loss_outputct) in (tuple, list) else loss_outputct
        lossmr = loss_outputmr[0] if type(loss_outputmr) in (tuple, list) else loss_outputmr
        #losstotal = loss *0.8+ lossct*0.1+ lossmr*0.1
        """
        if epoch < 10:
            losstotal = loss *0.2 + lossct*0.4 + lossmr*0.4
        elif epoch >=10 & epoch < 20:
            losstotal = loss *0.4 + lossct*0.3 + lossmr*0.3
        elif epoch >=20 & epoch < 30:
            losstotal = loss *0.6 + lossct*0.2+ lossmr*0.2
        elif epoch >=30 :
            losstotal = loss *0.8 + lossct*0.1+ lossmr*0.1
        """
        #losstotal = loss *0.5 + lossct + lossmr#loss+ lossct+ lossmr
        #losstotal = loss + lossct*0.5 + lossmr*0.5# 5 ZHE acc 70 auc 0.65 mcat wending lr-3 0.9
        #losstotal = loss*0.5 + lossct + lossmr#epoch 100 acc 65 auc 0.67 
        #losstotal = loss*0.8 + lossct + lossmr
        losstotal = loss+ lossct+ lossmr #acc 67.5 auc 0.66
        #losstotal = loss
        losses.append(losstotal.data[0])
        total_loss += losstotal.data[0]
        
        if select == 0:
            lossct.backward()
            optimizerCT.step()
        elif select == 1:
            lossmr.backward()
            optimizerMR.step()
        elif select == 2:
            losstotal.backward()
            optimizer.step()
        
        for metric in metrics:
            metric(outputs, target, loss_outputs)
        # if flag==True:
        #     print temp, outputs[0]
        #     acc, sens, spec, auc = accuracy(temp, outputs[0].data.max(1, keepdim=True)[1], 6, 0)
        #     print acc, sens, spec, auc
        #     print temp[0], outputs[0].data.max(1, keepdim=True)[1]
        #print(batch_idx) # only 11
        if (lengthdata *(batch_idx+1)) % log_interval == 0:# batch_idx % log_interval == 0 # batch_idx 0-11 ; len(data) 32
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ((batch_idx+1)*lengthdata), len(train_loader.dataset),#batch_idx * len(data[0]), len(train_loader.dataset),
                100.0*(lengthdata*(batch_idx+1)) / len(train_loader.dataset), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            #print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics,epoch,flag):
    Sens=[]
    Prec=[]
    F1=[]
    cnf_mat=[]
    for metric in metrics:
        metric.reset()
    model.eval()
    val_loss = 0
    for batch_idx, (dataCT, dataMR, target) in enumerate(val_loader):
        target = target if len(target) > 0 else None
        temp=target
        if not type(dataCT) in (tuple, list):
            data = (dataCT.cuda(),dataMR.cuda())
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        data = tuple(Variable(d, volatile=True) for d in data)

        outputs , CTout , MRout = model(*data)
        """
        _ , CTout , MRout = model(*data)
        outputs = CTout*0.5+MRout*0.5
        """
        #print("____________________")
        #print(target)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
            CTout = (CTout,)
            MRout = (MRout,)

        loss_inputs = outputs
        loss_inputct = CTout
        loss_inputmr = MRout
        if target is not None:
            target = Variable(target)
            target = (target,)
            loss_inputs += target
            loss_inputct += target
            loss_inputmr += target

        loss_outputs = loss_fn(*loss_inputs)
        loss_outputct = loss_fn(*loss_inputct)
        loss_outputmr = loss_fn(*loss_inputmr)
        
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        lossct = loss_outputct[0] if type(loss_outputct) in (tuple, list) else loss_outputct
        lossmr = loss_outputmr[0] if type(loss_outputmr) in (tuple, list) else loss_outputmr
        #losstotal = loss *0.8+ lossct*0.1+ lossmr*0.1
        """
        if epoch < 10:
            losstotal = loss *0.2 + lossct*0.4 + lossmr*0.4
        elif epoch >=10 & epoch < 20:
            losstotal = loss *0.4 + lossct*0.3 + lossmr*0.3
        elif epoch >=20 & epoch < 30:
            losstotal = loss *0.6 + lossct*0.2+ lossmr*0.2
        elif epoch >=30 :
            losstotal = loss *0.8 + lossct*0.1+ lossmr*0.1
        """
        losstotal = loss+ lossct+ lossmr
        #losstotal = loss

        val_loss += losstotal.data[0]

        for metric in metrics:
            metric(outputs, target, loss_outputs)
        if flag==True:
            #print temp[0:32],"--------------------------",temp[1],"--------------------------",outputs[0].data.max(1, keepdim=True)[1]
            l1, l2, l3, l4 = accuracy(temp, outputs[0].data.max(1, keepdim=True)[1],2, 0)
            Sens.append(l1)
            Prec.append(l2)
            F1.append(l3)
            cnf_mat.append(l4)
            #print(l4)
    #print acc,batch_idx
    return val_loss, metrics,Sens, Prec, F1, cnf_mat
