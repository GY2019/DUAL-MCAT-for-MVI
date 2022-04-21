# -*- coding:utf-8 -*-
from base.base_data_loader import BaseDataLoader

import os
import numpy as np
import scipy.io as sio
import torch
from torchvision import transforms
from SampleData import sampledata

class MRIORCT(BaseDataLoader):

    def __init__(self, config, selectnumber,train = True, transform=None, target_transform=None,classify = True,MRI = False):#f-ct t-mri
        super(MRIORCT, self).__init__(config)
        self.path = config.data_path
        self.model_type = config.exp_name 
        self.train = train
        self.isAug = config.isAug
        self.classify = classify
        self.MRI = MRI
        
        if config.classes == 5:   # 二分类：I、II  &  III、IV
            self.target_transform = self.binary
        else:
            self.target_transform = target_transform
            
        self.transform = self.ttd
        
        #提取指定模态数据并进行预处理
        if self.MRI == False:# CT
            self.dict = config.dictCT
            self.Fusion = config.FusionCT
        elif self.MRI == True:#MR
            self.dict = config.dictMR
            self.Fusion = config.FusionMR
        self.isTranspose = config.isTranspose   
        
        if self.train:
            if config.isSample:
                self.Num_train = config.Num_train
                (self.train_data, self.train_labels) = self.loadSampledData(self.train,selectnumber)
                print("load trainSample successful")
            else:
                (self.train_data, self.train_labels) = self.loadData(self.train,selectnumber)
                print("load train successful")
                
            self.train_data, self.train_labels = torch.from_numpy(self.train_data).float(),\
                                                 torch.from_numpy(self.train_labels).type(torch.LongTensor)
                                                 
        else:
            if config.isSample:
                self.Num_test = config.Num_test
                (self.test_data,self.test_labels) = self.loadSampledData(self.train,selectnumber)
                print("load testSample successful")
            else:
                (self.test_data, self.test_labels) = self.loadData(self.train,selectnumber)
                print("load test successful")
                
            self.test_data, self.test_labels = torch.from_numpy(self.test_data).float(),\
                                               torch.from_numpy(self.test_labels).type(torch.LongTensor)
        #print(train_labels)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img,target = self.train_data[index], self.train_labels[index]
        else:
            img,target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
        
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def ttd(self, img):
        dict = self.dict
        Fusion = self.Fusion
        img_list = []
        
        for modual in Fusion: # 3 channel only
            if len(Fusion[0]) == 1 or len(Fusion[0]) == 3:       
                img_list.append(img[:,:,dict[modual]].unsqueeze(2))
            else:
                img_list.append(img[:,:,dict[modual]])
                
        img = torch.stack(img_list, dim = 0)
        
        """
        for i in range(img.shape[2]):#15 all channel
            img_list.append(img[:,:,i].unsqueeze(2))
        img = torch.stack(img_list, dim = 0)
        """
        if self.isTranspose:
            # 转置
            img = img.permute(3, 0, 1, 2) #SxTxHxW
        else:
            img = img.permute(0, 3, 1, 2) #TxSxHxW
        
        # test(img)
        img = img.squeeze(0)
        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def binary(self, target):
        if target < 2:
            return 0
        else:
            return 1

    def getData(self, train=True, array=True):
        if train:
            if array:
                return self.train_data.numpy(), self.train_labels.numpy()
            else:
                return self.train_data, self.train_labels
        else:
            if array:
                return self.test_data.numpy(), self.test_labels.numpy()
            else:
                return self.test_data, self.test_labels
    
    def loadSampledData(self, train,selectnumber):
        path = str(self.path)
        data = []  
        labels = []
        successfulload = 0
        for line in selectnumber:
            if self.isAug and not(self.classify):
                Level = int(line[0])
                imgpath = line[2:]#-1
                imgpath = imgpath.replace('\r','')
                #print(imgpath)
		#print(line)
		#a=sio.loadmat("/home/lab304/xyj/Liver/Data/Binary/ABKEFGHIJ/1/00431620_1_0_ABKEFGHIJ_2_3_ud.mat")
		#print(a)
                npimg = np.load(imgpath.replace('\n',''))
                data.append(npimg)
                labels.append(Level)
                successfulload += 1
            else:
                if self.lineSearch(line, ['_90','_270','_180','_lr','ud','_T','_nT']):#not find only orignal
                    #print(line)
                    Level = int(line[0])
                    imgpath = line[2:]#-1
                    imgpath = imgpath.replace('\r','')
                    #print(imgpath)
                    npimg = np.load(imgpath.replace('\n',''))
                    data.append(npimg)
                    labels.append(Level)
                    successfulload += 1   
                    
        print("jia zai : "+str(successfulload))    
        labels = np.asarray(labels, dtype="float32")
        #print(labels)
        data = np.asarray(data) 
        return (data, labels)


    def loadData(self, train, selectnumber):
        path = str(self.path) 
        data = []  
        labels = []  
            
        successfulload = 0
        for line in selectnumber:
            if self.isAug:
                Level = int(line[0])
                imgpath = line[2:]
                imgpath = imgpath.replace('\r','')
                npimg = np.load(imgpath.replace('\n',''))
                data.append(npimg)
                labels.append(Level)
            else:
                if self.lineSearch(line, ['_90','_270','_180','_lr','ud','_T','_nT']):                
                    Level = int(line[0])
                    imgpath = line[2:]
                    imgpath = imgpath.replace('\r','')
                    npimg = np.load(imgpath.replace('\n',''))
                    data.append(npimg)
                    labels.append(Level)
            successfulload += 1
                    
	print("jia zai : "+str(successfulload))
        labels = np.asarray(labels, dtype="float32")
        data = np.asarray(data, dtype="float32") 
        # index = np.random.permutation(labels.shape[0])
        # X, y = data[index], labels[index];
        # return (X, y)
        return (data, labels)

    def lineSearch(self, line, strlist):
        for str in strlist:
            if str in line:
                return False
        return True

class MRIANDCT(BaseDataLoader):

    def __init__(self, config, selectnumber,train = True, transform=None, target_transform=None,classify = True):
        super(MRIANDCT, self).__init__(config)
        self.path = config.data_path
        self.classify = classify
        self.model_type = config.exp_name 
        self.train = train
        self.isAug = config.isAug
        
        if config.classes == 5:   # 二分类：I、II  &  III、IV
            self.target_transform = self.binary
        else:
            self.target_transform = target_transform
            
        self.transformCT = self.ttdCT
        self.transformMR = self.ttdMR
        
        #提取指定模态数据并进行预处理
        self.dictCT = config.dictCT
        self.FusionCT = config.FusionCT
        self.dictMR = config.dictMR
        self.FusionMR = config.FusionMR 
        
        self.isTranspose = config.isTranspose   
        
        if self.train:
            if config.isSample:
                self.Num_train = config.Num_train
                (self.train_CTdata,self.train_MRdata, self.train_labels) = self.loadSampledData(self.train,selectnumber)
                print("load trainSample successful")
            else:
                (self.train_CTdata,self.train_MRdata, self.train_labels) = self.loadData(self.train,selectnumber)
                print("load train successful")
                
            self.train_CTdata, self.train_MRdata, self.train_labels = torch.from_numpy(self.train_CTdata).float(),\
                                                 torch.from_numpy(self.train_MRdata).float(),\
                                                 torch.from_numpy(self.train_labels).type(torch.LongTensor)
        else:
            if config.isSample:
                self.Num_test = config.Num_test
                (self.test_CTdata, self.test_MRdata, self.test_labels) = self.loadSampledData(self.train,selectnumber)
                print("load testSample successful")
            else:
                (self.test_CTdata, self.test_MRdata, self.test_labels) = self.loadData(self.train,selectnumber)
                print("load test successful")
                
            self.test_CTdata, self.test_MRdata, self.test_labels = torch.from_numpy(self.test_CTdata).float(),\
                                               torch.from_numpy(self.test_MRdata).float(),\
                                               torch.from_numpy(self.test_labels).type(torch.LongTensor)
        #print(train_labels)
    def __getitem__(self, index):
        if self.train:
            imgCT, imgMR, target = self.train_CTdata[index], self.train_MRdata[index], self.train_labels[index]
        else:
            imgCT, imgMR, target = self.test_CTdata[index], self.test_MRdata[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        if self.transformCT is not None:
            imgCT = self.transformCT(imgCT)
        
        if self.transformMR is not None:
            imgMR = self.transformMR(imgMR)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return imgCT, imgMR, target
        
        
    def __len__(self):
        if self.train:
            return len(self.train_CTdata)
        else:
            return len(self.test_CTdata)

    def ttdCT(self, img):
        dict = self.dictCT
        Fusion = self.FusionCT
        img_list = []
        for modual in Fusion: 
            if len(Fusion[0]) == 1 or len(Fusion[0]) == 3:       
                img_list.append(img[:,:,dict[modual]].unsqueeze(2))
            else:
                img_list.append(img[:,:,dict[modual]])
                
        img = torch.stack(img_list, dim = 0)
        if self.isTranspose:
            # 转置
            img = img.permute(3, 0, 1, 2) #SxTxHxW
        else:
            img = img.permute(0, 3, 1, 2) #TxSxHxW
        
        # test(img)
        img = img.squeeze(0)
        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def ttdMR(self, img):
        dict = self.dictMR
        Fusion = self.FusionMR
        img_list = []
        for modual in Fusion: 
            if len(Fusion[0]) == 1 or len(Fusion[0]) == 3:       
                img_list.append(img[:,:,dict[modual]].unsqueeze(2))
            else:
                img_list.append(img[:,:,dict[modual]])
                
        img = torch.stack(img_list, dim = 0)
        if self.isTranspose:
            # 转置
            img = img.permute(3, 0, 1, 2) #SxTxHxW
        else:
            img = img.permute(0, 3, 1, 2) #TxSxHxW
        
        # test(img)
        img = img.squeeze(0)
        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img
        
    def binary(self, target):
        if target < 2:
            return 0
        else:
            return 1

    def getData(self, train=True, array=True):
        if train:
            if array:
                return self.train_data.numpy(), self.train_labels.numpy()
            else:
                return self.train_data, self.train_labels
        else:
            if array:
                return self.test_data.numpy(), self.test_labels.numpy()
            else:
                return self.test_data, self.test_labels
    
    def loadSampledData(self, train,selectnumber):
        path = str(self.path)              
            
        dataCT = []
        dataMR = []  
        labels = []   
            

	successfulload = 0
        for line in selectnumber:
            if self.isAug:
                Level = int(line[0])
                imgpath = line[2:]#-1
                imgpath = imgpath.replace('\r','')
                imgCTpath = imgpath.replace('MR2Nii','MR2Nii-CT')
                imgMRpath = imgpath.replace('MR2Nii','MR2Nii-MRI')
                #print(imgpath)
		#print(line)
		#a=sio.loadmat("/home/lab304/xyj/Liver/Data/Binary/ABKEFGHIJ/1/00431620_1_0_ABKEFGHIJ_2_3_ud.mat")
		#print(a)
                npimgCT = np.load(imgCTpath.replace('\n',''))
                dataCT.append(npimgCT)
                npimgMR = np.load(imgMRpath.replace('\n',''))
                dataMR.append(npimgMR)
                
                labels.append(Level)
                successfulload += 1
            else:
                if self.lineSearch(line, ['_90','_270','_180','_lr','ud','tr','tr2']):
                    Level = int(line[0])
                    imgpath = line[2:]#-1
                    imgpath = imgpath.replace('\r','')
                    imgCTpath = imgpath.replace('MR2Nii','MR2Nii-CT')
                    imgMRpath = imgpath.replace('MR2Nii','MR2Nii-MRI')
                    #print(imgpath)
                    npimgCT = np.load(imgCTpath.replace('\n',''))
                    dataCT.append(npimgCT)
                    npimgMR = np.load(imgMRpath.replace('\n',''))
                    dataMR.append(npimgMR)
                    
                    labels.append(Level)
                    successfulload += 1   
                    
        print("jia zai : "+str(successfulload))    
        labels = np.asarray(labels, dtype="float32")
        #print(labels)
        dataCT = np.asarray(dataCT) 
        dataMR = np.asarray(dataMR)
        return (dataCT, dataMR, labels)


    def loadData(self, train,selectnumber):
        path = str(self.path) 
        data = []  
        labels = []  
            
        successfulload = 0
        for line in selectnumber:
            if self.isAug:
                Level = int(line[0])
                imgpath = line[2:]
                imgpath = imgpath.replace('\r','')
                npimg = np.load(imgpath.replace('\n',''))
                data.append(npimg)
                labels.append(Level)
            else:
                if self.lineSearch(line, ['_90','_270','_180','_lr','ud','tr','tr2']):                
                    Level = int(line[0])
                    imgpath = line[2:]
                    imgpath = imgpath.replace('\r','')
                    npimg = np.load(imgpath.replace('\n',''))
                    data.append(npimg)
                    labels.append(Level)
            successfulload += 1
                    
            line = fp.readline()
        #print(labels)
        fp.close()
        print("jia zai : "+str(successfulload))
        labels = np.asarray(labels, dtype="float32")
        data = np.asarray(data, dtype="float32") 
        # index = np.random.permutation(labels.shape[0])
        # X, y = data[index], labels[index];
        # return (X, y)
        return (data, labels)

    def lineSearch(self, line, strlist):
        for str in strlist:
            if str in line:
                return False
        return True