# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:07:07 2019

@author: rain
"""
import numpy as np
import scipy.io as sio 
matfn=u'/home/lab304/xyj/Liver/Data/Binary/ABKEFGHIJ/1/00431620_1_0_ABKEFGHIJ_2_3_ud.mat'
data=sio.loadmat('/home/lab304/xyj/Liver/Data/Binary/ABKEFGHIJ/1/00431620_1_0_ABKEFGHIJ_2_3_ud.mat')
print(data['P'])
