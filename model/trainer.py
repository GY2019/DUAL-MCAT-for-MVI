from torch.autograd import Variable
import numpy as np
from utils.utils import modelsize
from utils1 import *

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, obj_label=False,flag=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
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
        scheduler.step()
        lr = scheduler.get_lr()
        Sens=[]
        Prec=[]
        F1=[]
        cnf_mat=[]
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, obj_label=obj_label,flag=flag)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}. lr:{}'.format(epoch + 1, n_epochs, train_loss,lr)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            temp1+=metric.value()
        val_loss, metrics,Sens, Prec, F1, cnf_mat = test_epoch(val_loader, model, loss_fn, cuda, metrics, obj_label=obj_label,flag=flag)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}. lr:{}'.format(epoch + 1, n_epochs,val_loss,lr)
        
                                                                        
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

def fit1(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, obj_label=False,flag=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
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
        #scheduler.step()
        lr = 1
        Sens=[]
        Prec=[]
        F1=[]
        cnf_mat=[]
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, obj_label=obj_label,flag=flag)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}. lr:{}'.format(epoch + 1, n_epochs, train_loss,lr)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            temp1+=metric.value()
        val_loss, metrics,Sens, Prec, F1, cnf_mat = test_epoch(val_loader, model, loss_fn, cuda, metrics, obj_label=obj_label,flag=flag)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}. lr:{}'.format(epoch + 1, n_epochs,val_loss,lr)
        
                                                                        
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            scheduler.step(metric.value())#step tiao zheng xue xi lv
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
    
def fit2(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, obj_label=False,flag=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()
    j=0
    
    temp=0.0
    temp1=0.0
    temp3=0.0
    Sens1=[]
    Prec1=[]
    F11=[]
    cnf_mat1=[]

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        lr = scheduler.get_lr()
        Sens=[]
        Prec=[]
        F1=[]
        cnf_mat=[]
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, obj_label=obj_label,flag=flag)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}. lr:{}'.format(epoch + 1, n_epochs, train_loss,lr)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            temp1+=metric.value()
        val_loss, metrics,Sens, Prec, F1, cnf_mat = test_epoch(val_loader, model, loss_fn, cuda, metrics, obj_label=obj_label,flag=flag)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}. lr:{}'.format(epoch + 1, n_epochs,val_loss,lr)
        
                                                                        
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            temp+=metric.value()
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
    return temp1/n_epochs,temp/n_epochs,temp3,Sens1, Prec1, F11, cnf_mat1
        
def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, obj_label,flag):
    acc = []
    sens = []
    spec = []
    auc = []

    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    if obj_label:
        for batch_idx, (data, target, obj_l) in enumerate(train_loader):
            #print("____________________")
            #print(target)
            lengthdata = len(data)# additional
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            data = tuple(Variable(d) for d in data)

            optimizer.zero_grad()
            outputs = model(*data)
            
            # modelsize(model, *data)
            # # print(outputs)
            # raw_input()
            #print("____________________")
            #print(target)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = Variable(target)
                target = (target,)
                loss_inputs += target
            print(loss_inputs)   
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.data[0])
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(outputs, target, loss_outputs)
            
            # if flag==True:
            #     #acc, sens, spec, auc = accuracy(temp, outputs, 6, 0)
            #     #print acc, sens, spec, auc
            #     print temp[0], outputs[0].data.max(1, keepdim=True)[1]
            if (lengthdata *(batch_idx+1)) % log_interval == 0:# batch_idx % log_interval == 0 # batch_idx 0-11 ; len(data) 32
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ((batch_idx+1)*lengthdata), len(train_loader.dataset),#batch_idx * len(data[0]), len(train_loader.dataset),
                    100.0*(lengthdata*(batch_idx+1)) / len(train_loader.dataset), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                #print(message)
                losses = []

        total_loss /= (batch_idx + 1)    
    else:
        for batch_idx, (data, target) in enumerate(train_loader):
            #print("____________________")
            #print(len(train_loader)) #12  = 384 /32
            lengthdata = len(data)# additional
            temp=target
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            data = tuple(Variable(d) for d in data)

            optimizer.zero_grad()
            outputs = model(*data)
            #print("____________________")
            #print(target)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = Variable(target)
                target = (target,)
                loss_inputs += target
            #print(loss_inputs)
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.data[0])
            total_loss += loss.data[0]
            loss.backward()
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

def test_epoch(val_loader, model, loss_fn, cuda, metrics, obj_label,flag):
    Sens=[]
    Prec=[]
    F1=[]
    cnf_mat=[]
    for metric in metrics:
        metric.reset()
    model.eval()
    val_loss = 0
    if obj_label:
        for batch_idx, (data, target, obj_l) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            temp=target
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            data = tuple(Variable(d, volatile=True) for d in data)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = Variable(target, volatile=True)
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.data[0]

            for metric in metrics:
                metric(outputs, target, loss_outputs)
    else:
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            temp=target
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            data = tuple(Variable(d, volatile=True) for d in data)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = Variable(target, volatile=True)
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.data[0]

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
