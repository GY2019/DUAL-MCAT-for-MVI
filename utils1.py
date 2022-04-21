# -*- coding:utf-8 -*-
import os
import csv
import numpy as np
import scipy.io as sio
from scipy import interp
import itertools
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
#from keras import backend as K  

#import tensorflow as tf 
from PIL import Image
#from keras.utils import np_utils    
#from keras.utils.vis_utils import plot_model 

def loadData(path):  
    data = []  
    labels = []  
    for i in range(2):  
        dir = './'+path+'/'+str(i)  
        list = os.listdir(dir)
        for img in list:   
            mat = sio.loadmat(dir+'/'+img)
            # print mat.keys()
            data.append(mat['P'])
            labels.append(i)
        print path, i, 'is read'
    labels = np_utils.to_categorical(labels, 2)
    data = np.asarray(data, dtype="float32")
    print data.shape
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    return (X_train, y_train), (X_test, y_test)

    
def loadSplitData(path, classes):  
    train_data = []  
    train_labels = []  
    fp = open(os.path.join(path, "train.txt"), 'r')
    line = fp.readline()   # 调用文件的 readline()  
    while len(line):
        # print '*'*10,line      
        Level = int(line[0])
        imgpath = line[2:-1]
        mat = sio.loadmat(imgpath)
        # print mat.keys()
        train_data.append(mat['P'])
        train_labels.append(Level)
        line = fp.readline()
    fp.close()
    train_labels = np_utils.to_categorical(train_labels, classes)
    train_data = np.asarray(train_data, dtype="float32")
    print train_data.shape 
    # 打乱顺序
    index = np.random.permutation(train_labels.shape[0])
    X_train, y_train = train_data[index], train_labels[index];   

    test_data = []  
    test_labels = []  
    fp = open(os.path.join(path, "test.txt"), 'r')
    line = fp.readline()   # 调用文件的 readline()  
    while len(line):
        # print '*'*10,line      
        Level = int(line[0])
        imgpath = line[2:-1]
        mat = sio.loadmat(imgpath)
        # print mat.keys()
        test_data.append(mat['P'])
        test_labels.append(Level)
        line = fp.readline()
    fp.close()
    test_labels = np_utils.to_categorical(test_labels, classes)
    test_data = np.asarray(test_data, dtype="float32")
    print test_data.shape
    # 打乱顺序
    index = np.random.permutation(test_labels.shape[0])
    X_test, y_test = test_data[index], test_labels[index];
    
    return (X_train, y_train), (X_test, y_test)
    
def combData(path, C, dict, isTrans):
    (trainData, trainLabels), (testData, testLabels) = loadSplitData(path, 2)
    classes = 2
    Train_list = []
    Test_list = []    
    for modual in C: 
        if len(modual) == 1 or len(modual) == 3:  
            input_shape = (trainData.shape[1], trainData.shape[2])
            Train_list.append(trainData[:,:,:,dict[modual]].reshape(trainData.shape[0], input_shape[0], input_shape[1]))
            Test_list.append(testData[:,:,:,dict[modual]].reshape(testData.shape[0], input_shape[0], input_shape[1])) 
        else:  
            input_shape = (trainData.shape[1], trainData.shape[2], 5)
            Train_list.append(trainData[:,:,:,dict[modual]].reshape(trainData.shape[0], input_shape[0], input_shape[1], 5))
            Test_list.append(testData[:,:,:,dict[modual]].reshape(testData.shape[0], input_shape[0], input_shape[1], 5))
       
    if len(C[0]) == 1 or len(C[0]) == 3:         
        X_train = tf.cast(tf.stack(Train_list, axis=3), tf.float32)
        X_test = tf.cast(tf.stack(Test_list, axis=3), tf.float32)
    else:
        X_train = tf.cast(tf.stack(Train_list, axis=4), tf.float32)
        X_test = tf.cast(tf.stack(Test_list, axis=4), tf.float32)
        if isTrans:
            # # 转置
            X_train = tf.transpose(X_train, perm=[0,1,2,4,3])
            X_test = tf.transpose(X_test, perm=[0,1,2,4,3])
  
    with tf.Session() as sess:
        X_train, X_test = X_train.eval(), X_test.eval()    
    
    print X_train.shape, trainLabels.shape 
    # raw_input()    
    return (X_train, trainLabels), (X_test, testLabels), classes    

def accuracy_curve(h, dataset, isSample, save_tag):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    
    # # 绘图
    plt.figure(figsize=(17, 5))
    plt.subplot(211)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend(loc = 'lower right')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend(loc = 'upper right')
    plt.grid(True)
    plt.savefig('img/'+ save_tag + '_acc.png')
    # plt.show()
    plt.close('all') # 关闭图
    
    #python2可以用file替代open
    # save_file = dataset + '.txt'
    # f = open(save_file,'ab')
    # f.write('sample:'+str(isSample)+'\n')
    # f.write('acc:'+str(acc)+'\n')
    # f.write('val_acc:'+str(val_acc)+'\n')
    
    # f.write('loss:'+str(loss)+'\n')
    # f.write('val_loss:'+str(val_loss)+'\n\n')
    # f.close()   

def plot_confusion_matrix(cm, classes,
                          save_tag = '',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('img/'+ save_tag + '_cfm.png')
    plt.close('all') # 关闭图    
      
def accuracy(y_true, y_pred, classes, isPlot, save_tag = ''):        
    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))    
    for i in range(len(y_true)): 
        #print y[i],"--------------------",y_true[i]
        y[i] = y_true[i]
        y_[i] = y_pred[i]
    #print y,y_
    cnf_mat = confusion_matrix(y, y_, labels=range(0,classes))#
    #print cnf_mat.shape
    
    if classes > 1: 
        if isPlot:
            # # 绘制混淆矩阵
            plot_confusion_matrix(cnf_mat, range(2), save_tag=save_tag)   
        # 计算多分类评价值
        Sens = recall_score(y, y_, average='macro')
        Prec = precision_score(y, y_, average='macro')
        F1 = f1_score(y, y_, average='weighted') 
        Support = precision_recall_fscore_support(y, y_, beta=0.5, average=None)
        #print Support
        return Sens, Prec, F1, cnf_mat
    else:
        Acc = 1.0*(cnf_mat[1][1]+cnf_mat[0][0])/len(y_true)
        Sens = 1.0*cnf_mat[1][1]/(cnf_mat[1][1]+cnf_mat[1][0])
        Spec = 1.0*cnf_mat[0][0]/(cnf_mat[0][0]+cnf_mat[0][1])
        if isPlot:
            # # 绘制混淆矩阵
            plot_confusion_matrix(cnf_mat, range(classes), save_tag=save_tag)
            # # 绘制ROC曲线
            fpr, tpr, thresholds = roc_curve(y_true[:,1], y_pred[:,1])
            fpr[0], tpr[0] = 0, 0
            fpr[-1], tpr[-1] = 1, 1

            Auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (Auc))
            
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color=(0.6, 0.6, 0.6), alpha=.8)
            
            plt.xlim([-0.05, 1.05])  
            plt.ylim([-0.05, 1.05])  
            plt.xlabel('False Positive Rate')  
            plt.ylabel('True Positive Rate')  
            plt.title('Receiver operating characteristic example')  
            plt.legend(loc="lower right") 
            plt.savefig('img/'+ save_tag + '_roc.png')
            # plt.show()
            plt.close('all') # 关闭图
            
            # # 记录ROC曲线以及曲线下面积           
            f = open('img/roc_record.txt', 'ab+')
            f.write(save_tag + 'AUC:' +  str(Auc) + '\n')
            f.write('FPR:' + str(list(fpr)) + '\n')
            f.write('TPR:' + str(list(tpr)) + '\n\n')
            f.close()
            
            # #字典中的key值即为csv中列名
            # dataframe = pd.DataFrame({'FPR':fpr,'TPR':tpr})
            # #将DataFrame存储为csv,index表示是否显示行名，default=True
            # dataframe.to_csv('img/roc_record.csv', index=False, sep=',')            
            
        # 计算AUC值
        Auc = roc_auc_score(y_true[:,1], y_pred[:,1])
        return Acc, Sens, Spec, Auc 