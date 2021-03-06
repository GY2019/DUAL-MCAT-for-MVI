#-*- coding:utf-8 -*-
import os
import re
import pandas
import pickle
import numpy as np
from PIL import Image
import scipy.io as sio
import scipy.misc as misc

BBOX_DATA_DIR = '../../Data/BboxAug'
MAT_DATA_DIR = '/media/lab304/J52/Dataset/Mat'
LABEL_PATH = '../../Data/labels.csv'
SAVE_DIR = '../../Data'

sampleSet = set()

def readLabel(asIndex = 'modalNo', index = 'A'):
    '''Read labels
        Read labels as request.

    Args:
        asIndex: String, Index type.
        index: String, Index.

    Returns:
        labels: A DataFrame of reorganized labels.
        For example:

                            serNo                 Location        Center meanSize  \
        tumourNo modalNo                                                          
        1        A            7    [257,167,301,236,5,8]   [279,201,7]  [42,64]   
                 B            8    [255,178,300,237,6,8]   [277,207,8]  [42,64]   
                              ...
                 J           22  [247,170,287,234,14,24]  [267,202,22]  [42,64]   
        2        A            8    [214,182,254,229,6,9]   [234,205,8]  [35,42]   
                 B            8    [218,183,253,227,7,9]   [235,205,8]  [35,42]   
                              ...
                 J           25  [206,186,240,227,23,28]  [223,206,25]  [35,42]   
        3        A            9    [281,142,303,166,8,9]   [292,154,9]  [19,22]   
                              ... 

        Or:
                              serNo                 Location        Center   meanSize  \
        patientNo  tumourNo                                                            
        '00070993' 1             8   [223,176,284,242,6,10]   [253,209,8]    [62,65]   
        '00090960' 1             5    [191,139,224,184,5,5]   [207,161,5]    [31,42]   
        '00159253' 1            13  [231,149,288,206,11,15]  [259,177,13]    [56,57]   
        '00190415' 1             7    [257,167,301,236,5,8]   [279,201,7]    [42,64]   
                   2             8    [214,182,254,229,6,9]   [234,205,8]    [35,42]   
                   3             9    [281,142,303,166,8,9]   [292,154,9]    [19,22]   
        '00431620' 1            15    [245,64,348,172,9,17]  [296,118,15]   [97,103]   
        '00525849' 1             9    [153,216,183,242,9,9]   [168,229,9]    [28,25]   
        '00582685' 1            12  [264,104,317,159,10,13]  [290,131,12]    [56,59] 
                               ...        

    Raises:
        None
    Usage:
        readLabel(asIndex = 'patientNo', index = '00190415')  
        readLabel(asIndex = 'modalNo', index = 'A') 
    '''
    #???csv??????
    if asIndex == 'modalNo':
        labels = pandas.read_csv(LABEL_PATH, index_col=[2,0,1])
    elif asIndex == 'patientNo':
        index = '\'' + index + '\''
        labels = pandas.read_csv(LABEL_PATH, index_col=[0,1,2])
    labels = labels.fillna('null')
    ''' DataFeame usage:
    # print(labels.dtypes)
    # print(labels.iloc[0])
    # print(labels.loc[('\'00190415\'', 2, 'B'), :])
    # print(labels.loc[('\'00190415\'', 2, 'B'), 'WHO'])
    # print(labels.loc['\'00190415\'', 2, 'B']['WHO'])
    '''
    return labels.loc[index]

def readBbox(liverVolume, tumourInfo, saveNeg=False):
    pattern = re.compile(r'[\d]')
    tumourLoc = [int(x) for x in tumourInfo['Location'][1:-1].split(',') if pattern.search(x)]
    tumourCenter = [int(x) for x in tumourInfo['Center'][1:-1].split(',') if pattern.search(x)]
    tumourSize = [int(x) for x in tumourInfo['meanSize'][1:-1].split(',') if pattern.search(x)]
    if saveNeg:
        tumourD = [int(x) for x in tumourInfo['d'][1:-1].split(',') if pattern.search(x)]
        tumourCenter[0], tumourCenter[1] = tumourCenter[0] + tumourD[0], tumourCenter[1] + tumourD[1]
        
    return liverVolume[tumourCenter[0]-tumourSize[0]/2:tumourCenter[0]+tumourSize[0]/2+1,
                       tumourCenter[1]-tumourSize[1]/2:tumourCenter[1]+tumourSize[1]/2+1,
                       tumourLoc[4]:tumourLoc[5]+1]

def saveSliceToFile(Bbox, index, savePath):
    '''
    # Method 1?????????numpy???rot90???flip
    '''
    
    # ????????????
    image = np.squeeze(Bbox[:, :, index])
    misc.imsave(savePath + '.png', image)   

    # ???????????????        
    ############## rotate volume ##############
    rot90 = np.rot90(Bbox)                  #?????????????????????90
    image = np.squeeze(rot90[:, :, index])
    misc.imsave(savePath + '_90.jpg', image) 
    
    rot180 = np.rot90(Bbox, 2)              #?????????????????????180
    image = np.squeeze(rot180[:, :, index])
    misc.imsave(savePath + '_180.jpg', image)
    
    rot270 = np.rot90(Bbox, 3)              #?????????????????????270
    image = np.squeeze(rot270[:, :, index])
    misc.imsave(savePath + '_270.jpg', image)
    
    ############### flip volume ###############
    lr = np.fliplr(Bbox)                    #????????????
    image = np.squeeze(lr[:, :, index])
    misc.imsave(savePath + '_lr.jpg', image)    
    
    ud = np.flipud(Bbox)                    #????????????
    image = np.squeeze(ud[:, :, index])
    misc.imsave(savePath + '_ud.jpg', image)    

    print(savePath+' saved!') 
    
    '''
    # Method 2: ??????PIL Image???transpose??????rotate
    # ???????????????????????????????????????????????????
    roi = Bbox[:, :, index]
    # ??????
    image = np.squeeze(roi)
    image = Image.fromarray(image)
    
    # ???????????????????????????????????????
    w, h = image.size
    m = min(w, h)
    if m < 32:
        w, h = int(32.0*w/m), int(32.0*h/m)
        image = image.resize((w, h))  #???????????????
        print(w, h)

    # ????????????
    misc.imsave(savePath + '.png', roi)
    
    # ???????????????
    # dst5 = img.rotate(45)                  #???????????????45
    img = image.transpose(Image.ROTATE_90)   #?????????????????????90
    misc.imsave(savePath + '_90.jpg', img)
    img = image.transpose(Image.ROTATE_180)  #?????????????????????180
    misc.imsave(savePath + '_180.jpg', img)
    img = image.transpose(Image.ROTATE_270)  #?????????????????????270
    misc.imsave(savePath + '_270.jpg', img)
    img = image.transpose(Image.FLIP_LEFT_RIGHT)  #????????????
    misc.imsave(savePath + '_lr.jpg', img)
    img = image.transpose(Image.FLIP_TOP_BOTTOM)  #????????????
    misc.imsave(savePath + '_ud.jpg', img)
    print(savePath+' saved!')  
    '''
    
def saveFusionToFile(picMat, savePath, saveTpye):
    if saveTpye == '.png':
        # ????????????
        misc.imsave(savePath+saveTpye, picMat)
        
        # ???????????????        
        ############## rotate volume ##############
        rot90 = np.rot90(picMat)              #?????????????????????90
        misc.imsave(savePath + '_90' + saveTpye, rot90)
        
        rot180 = np.rot90(picMat, 2)          #?????????????????????180
        misc.imsave(savePath + '_180' + saveTpye, rot180) 

        rot270 = np.rot90(picMat, 3)          #?????????????????????270
        misc.imsave(savePath + '_270' + saveTpye, rot270)   
        
        ############### flip volume ###############
        lr = np.fliplr(picMat)                #????????????
        misc.imsave(savePath + '_lr' + saveTpye, lr)    
    
        ud = np.flipud(picMat)                #????????????
        misc.imsave(savePath + '_ud' + saveTpye, ud)
        
    else:
        # ????????????
        np.save(savePath+saveTpye, picMat)
  
        # ???????????????        
        ############## rotate volume ##############
        rot90 = np.rot90(picMat)              #?????????????????????90
        np.save(savePath + '_90' + saveTpye, rot90)
        
        rot180 = np.rot90(picMat, 2)          #?????????????????????180
        np.save(savePath + '_180' + saveTpye, rot180) 

        rot270 = np.rot90(picMat, 3)          #?????????????????????270
        np.save(savePath + '_270' + saveTpye, rot270)   
        
        ############### flip volume ###############
        lr = np.fliplr(picMat)                #????????????
        np.save(savePath + '_lr' + saveTpye, lr)    
    
        ud = np.flipud(picMat)                #????????????
        np.save(savePath + '_ud' + saveTpye, ud)

    print(savePath+' saved!') 
    
def saveSlice(patientNo, tumourNo, modal, Bbox, tumourInfo, standard, saveNeg=False):
    if saveNeg and patientNo in ['00431620', '03930451']:
        return
    pattern = re.compile(r'[\d]')
    tumourLoc = [int(x) for x in tumourInfo['Location'][1:-1].split(',') if pattern.search(x)]
    serNo = tumourInfo['serNo']
    tumourWHO = tumourInfo['WHO']
    tumourEd = int(tumourInfo['Edmondson'])
    # ??????????????????
    saveDir = os.path.join(os.path.join(SAVE_DIR, standard), modal)
    if standard == 'Binary':
        if saveNeg:
            saveDir = os.path.join(saveDir, '0')
        else:
            saveDir = os.path.join(saveDir, '1')
    else:
        if standard == 'WHO':
            saveDir = os.path.join(saveDir, str(tumourWHO-1))
        else:
            saveDir = os.path.join(saveDir, str(tumourEd-1))
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
        
    up, bottom = serNo-tumourLoc[4], tumourLoc[5]-serNo
    up, bottom = int(up*0.75+0.5), int(bottom*0.75+0.5)
    
    # ??????????????????
    sampleName = patientNo + '_' + str(tumourNo+1) + '_0_' + modal\
                           + '_' + str(tumourWHO) + '_' + str(tumourEd)
    savePath = os.path.join(saveDir, sampleName)
    saveSliceToFile(Bbox, up, savePath)# ???????????????
    sampleSet.add(patientNo + '_' + str(tumourNo+1) + '_0')# ??????????????????
    
    # ??????????????????
    for i in range(up):
        sampleName = patientNo + '_' + str(tumourNo+1) + '_' + str(-1-i) + '_' + modal\
                               + '_' + str(tumourWHO) + '_' + str(tumourEd)
        savePath = os.path.join(saveDir, sampleName)
        saveSliceToFile(Bbox, up-i-1, savePath)# ???????????????
        sampleSet.add(patientNo + '_' + str(tumourNo+1) + '_' + str(-1-i))
        
    # ??????????????????
    for i in range(bottom):
        sampleName = patientNo + '_' + str(tumourNo+1) + '_' +str(i+1) +  '_' + modal\
                               + '_' + str(tumourWHO) + '_' + str(tumourEd)
        savePath = os.path.join(saveDir, sampleName)
        saveSliceToFile(Bbox, up+i+1, savePath)# ???????????????
        sampleSet.add(patientNo + '_' + str(tumourNo+1) + '_' +str(i+1))
        
def saveFusion(patientNo, tumourNo, Bboxs, BboxInfo, fusionName, standard, saveTpye, saveNeg=False):
    global sampleSet
    if saveNeg and patientNo in ['00431620', '03930451']:
        return
    tumourWHO = BboxInfo[0]['WHO']
    tumourEd = int(BboxInfo[0]['Edmondson'])
    # ??????????????????
    saveDir = os.path.join(os.path.join(SAVE_DIR, standard), fusionName)
    if standard == 'Binary':
        if saveNeg:
            saveDir = os.path.join(saveDir, '0')
        else:
            saveDir = os.path.join(saveDir, '1')
    else:
        if standard == 'WHO':
            saveDir = os.path.join(saveDir, str(tumourWHO-1))
        else:
            saveDir = os.path.join(saveDir, str(tumourEd-1))
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    tumourSize = Bboxs[0].shape
    # ??????????????????
    picMat = np.zeros((tumourSize[0], tumourSize[1], len(BboxInfo)))
    
    # ????????????????????????????????????????????????
    upSlice, bottomSlice = [], [] #????????????????????????(??????????????????????????????????????????), ???????????????????????????
    for info in BboxInfo:
        tumourLoc = [int(x) for x in info['Location'][1:-1].split(',')]
        serNo, Z1, Z2 = info['serNo'], tumourLoc[4], tumourLoc[5]
        upSlice.append(serNo-Z1)
        bottomSlice.append(Z2-serNo)
    up, bottom = min(upSlice), min(bottomSlice)
    print(upSlice, bottomSlice, up, bottom, int(up*0.75+0.5), int(bottom*0.75+0.5))
    # ???????????????????????????
    up, bottom = int(up*0.75+0.5), int(bottom*0.75+0.5)
    # ??????????????????
    for index, info in enumerate(BboxInfo):
        picMat[:, :, index] = Bboxs[index][:, :, upSlice[index]]
    sampleName = patientNo + '_' + str(tumourNo+1) + '_0_' + fusionName\
                           + '_' + str(tumourWHO) + '_' + str(tumourEd)
    savePath = os.path.join(saveDir, sampleName)
    saveFusionToFile(picMat, savePath, saveTpye)
    sampleSet.add(patientNo + '_' + str(tumourNo+1) + '_0')
    
    # ??????????????????
    for i in range(up):
        for index, info in enumerate(BboxInfo):
            picMat[:, :, index] = Bboxs[index][:, :, upSlice[index]-i-1]
        sampleName = patientNo + '_' + str(tumourNo+1) + '_' +str(-1-i) +  '_' + fusionName\
                               + '_' + str(tumourWHO) + '_' + str(tumourEd)
        savePath = os.path.join(saveDir, sampleName)
        saveFusionToFile(picMat, savePath, saveTpye)
        sampleSet.add(patientNo + '_' + str(tumourNo+1) + '_' +str(-1-i))
        
    # ??????????????????
    for i in range(bottom):
        for index, info in enumerate(BboxInfo):
            picMat[:, :, index] = Bboxs[index][:, :, upSlice[index]+i+1]
        sampleName = patientNo + '_' + str(tumourNo+1) + '_' + str(i+1) + '_' + fusionName\
                               + '_' + str(tumourWHO) + '_' + str(tumourEd)
        savePath = os.path.join(saveDir, sampleName)
        saveFusionToFile(picMat, savePath, saveTpye)
        sampleSet.add(patientNo + '_' + str(tumourNo+1) + '_' + str(i+1))
    
def readModalData(modal = 'A', standard = 'WHO'):
    '''
    ????????????????????????
    '''   
    if modal == 'K':
        labels = readLabel(asIndex = 'modalNo', index = 'B')
    else:
        labels = readLabel(asIndex = 'modalNo', index = modal)
    patientList = os.listdir(MAT_DATA_DIR)
    for patientNo in patientList:
        # ??????MRI????????? 512x512xS
        dataDir = os.path.join(MAT_DATA_DIR, patientNo)
        dataPath = os.path.join(dataDir, modal+'.mat')
        liverVolume = sio.loadmat(dataPath)['D']
        # ??????tumour??????
        patientInfo = labels.loc['\'' + patientNo + '\'']
        for tumourNo in range(len(patientInfo)):
            # print(patientNo, tumourNo)
            # ??????????????????
            tumourInfo = patientInfo.iloc[tumourNo]
            # roi??????
            posBbox = readBbox(liverVolume, tumourInfo)
            saveSlice(patientNo, tumourNo, modal, posBbox, tumourInfo, standard)
            # print(posBbox.shape)
            
            if standard is 'Binary':
                # ????????????
                negBbox = readBbox(liverVolume, tumourInfo, saveNeg=True)
                saveSlice(patientNo, tumourNo, modal, negBbox, tumourInfo, standard, saveNeg=True)
            
def readPatientData(Fusion = ['A', 'B', 'K'], standard = 'WHO', saveTpye = '.png'):
    '''
    ????????????????????????????????????
    '''
    patientList = os.listdir(MAT_DATA_DIR)
    for patientNo in patientList:
        # ???????????????????????????
        labels = readLabel(asIndex = 'patientNo', index = patientNo)
        for tumourNo in range(len(labels)/8):
            Info = labels.loc[tumourNo+1]
            posBboxs, negBboxs, BboxInfo, fusionName = [], [], [], ''
            for modal in Fusion:
                # ??????MRI????????? 512x512xS
                fusionName = fusionName + modal
                dataDir = os.path.join(MAT_DATA_DIR, patientNo)
                dataPath = os.path.join(dataDir, modal+'.mat')
                liverVolume = sio.loadmat(dataPath)['D']
                if modal == 'K':
                    tumourInfo = Info.loc['B']
                else:
                    tumourInfo = Info.loc[modal]            
                # ??????Bbox
                posBbox = readBbox(liverVolume, tumourInfo)
                posBboxs.append(posBbox)
                BboxInfo.append(tumourInfo)
                if standard == 'Binary':
                    negBbox = readBbox(liverVolume, tumourInfo, saveNeg=True) 
                    negBboxs.append(negBbox)                    
            
            saveFusion(patientNo, tumourNo, posBboxs, BboxInfo, fusionName,
                                    standard, saveTpye)
            if standard == 'Binary':
                saveFusion(patientNo, tumourNo, negBboxs, BboxInfo, fusionName,
                                        standard, saveTpye, saveNeg=True)
        
def main():
    # modalList = ['A', 'B', 'K', 'E', 'F', 'G', 'H', 'I', 'J']
    fusionList = ['ABK', 'EGJ']
    # ??????????????????
    # for modal in modalList:
        # readModalData(modal=modal, standard = 'WHO')
        
    # # ??????????????????
    # for fusion in fusionList:
        # readPatientData(Fusion = list(fusion), standard = 'WHO')
    
    # # ??????????????????
    # for modal in modalList:
        # readModalData(modal=modal, standard = 'Edmondson')
        
    # # ??????????????????
    # for fusion in fusionList:
        # readPatientData(Fusion = list(fusion), standard = 'Edmondson')
        
        # # ??????????????????
    # for modal in modalList:
        # readModalData(modal=modal, standard = 'Binary')
        
    # ??????????????????
    for fusion in fusionList:
        readPatientData(Fusion = list(fusion), standard = 'Binary')
      
    ########################################################################################      
    fusionList = ['EFGHIJ', 'ABKEFGHIJ'] 
    global sampleSet 
    # ??????????????????
    for fusion in fusionList:
        sampleSet.clear()
        readPatientData(Fusion = list(fusion), standard = 'WHO', saveTpye = '.npy')
        binPath = os.path.join(os.path.join(os.path.join(SAVE_DIR, 'WHO'), fusion), 'sample.bin')
        with open(binPath, 'wb') as fp:
            pickle.dump(sampleSet, fp)
            
        # with open(binPath, 'rb') as fp:
            # data=pickle.load(fp)#??????????????????
            # print(data)

    for fusion in fusionList:
        sampleSet.clear()
        readPatientData(Fusion = list(fusion), standard = 'Edmondson', saveTpye = '.npy')
        binPath = os.path.join(os.path.join(os.path.join(SAVE_DIR, 'Edmondson'), fusion), 'sample.bin')
        with open(binPath, 'wb') as fp:
            pickle.dump(sampleSet, fp)

    for fusion in fusionList:
        sampleSet.clear()
        readPatientData(Fusion = list(fusion), standard = 'Binary', saveTpye = '.npy')
        binPath = os.path.join(os.path.join(os.path.join(SAVE_DIR, 'Binary'), fusion), 'sample.bin')
        with open(binPath, 'wb') as fp:
            pickle.dump(sampleSet, fp)
  
if __name__ == "__main__":
    main()