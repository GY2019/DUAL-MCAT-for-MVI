# coding=gbk
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import pickle
import random
from collections import defaultdict
def DataAug(labellist):
    Aug_operation=["","_lr","_ud","_90","_180","_270","_T","_nT"]
    data_Aug=[]
    for dataline in labellist:
        for element in Aug_operation:
            temp=dataline.replace('.npy','')+element+".npy"
            data_Aug.append(temp)
    random.shuffle(data_Aug)
    return np.array(data_Aug)
    
    
def resamplelabel(labellist):
    datasample = []
    labels = [int(x[0]) for x in labellist]
    label_linenums = defaultdict(list)
    for i, label in enumerate(labels):
        label_linenums[label] += [i]
    #print(label)
    #print(len(labels))
    #print(label_linenums)
    
    ret = []
     
    # classes with fewer data are sampled first;
    label_list = sorted(label_linenums, key=lambda x: len(label_linenums[x]))
    min_class = label_list[0]
    maj_class = label_list[-1]
    min_class_num = len(label_linenums[min_class])
    maj_class_num = len(label_linenums[maj_class])
    
    
    print("min class:"+str(min_class))
    print("max class:"+str(maj_class))
    print("min class mun:"+str(min_class_num))
    print("max class mun:"+str(maj_class_num))
    
    method=1 # 
    subset_size = 20
    """
         0 -- over-sampling & under-sampling given subclass_size
         1 -- over-sampling (subclass_size: any value)
         2 -- under-sampling(subclass_size: any value)
    """
    random_state = check_random_state(42)
    for label in label_list:
        linenums = label_linenums[label]
        label_size = len(linenums)
        if  method == 0:
            if label_size<subset_size:
                ret += linenums
                subnum = subset_size-label_size
            else:
                subnum = subset_size
            ret += [linenums[i] for i in random_state.randint(low=0, high=label_size,size=subnum)]
        elif method == 1:
            if label == maj_class:
                ret += linenums
                continue
            else:
                ret += linenums
                subnum = maj_class_num-label_size               
                ret += [linenums[i] for i in random_state.randint(low=0, high=label_size,size=subnum)]
        elif method == 2:
            if label == min_class:
                ret += linenums
                continue
            else:
                subnum = min_class_num
                ret += [linenums[i] for i in random_state.randint(low=0, high=label_size,size=subnum)]
    random.shuffle(ret)
    
    for i in ret:
        datasample.append(labellist[i])
    print(len(datasample))
    #print(len(ret))
    return np.array(datasample)
