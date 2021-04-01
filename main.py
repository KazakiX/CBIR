import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, measure
from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm

import cv2
import fast_glcm


def colorsearch(imgpath):
    Hist = {}
    results = {}
    src =[]
    imgCV = cv2.imread(imgpath)
    #self.testImg为待匹配图片
    testHist = cv2.calcHist([imgCV],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    #提取直方图
    testHist = cv2.normalize(testHist,testHist,0,255,cv2.NORM_MINMAX).flatten()
    with open("color.csv") as f:
        reader = csv.reader(f)
            # loop over the rows in the index
        for row in reader:
            # parse out the imageID and features,
            # then compute the chi-squared distance
            features = [float(x) for x in row[1:]]
            Hist[row[0]]=features
    # print(type(np.array(Hist["F:\CBIR\\tongyi\\accordion\\image_0001.jpg"])))
    
    for (k,hist) in Hist.items():
        d = 0.5*np.sum([((a-b)**2)/(a+b+1e-10) for(a,b) in zip(hist,testHist)])
        results[k] = d
    results = sorted([(v, k) for (k, v) in results.items()])
    i = 1
    plt.figure('colorhist')
    testimg = plt.imread(imgpath)
    plt.subplot(3,4,1)
    plt.imshow(testimg)
    plt.axis('off')
    plt.title('testimg')
    for k,v in results:
        if i < 11:
            # print(v)
            img = plt.imread(v)
            plt.subplot(3,4,i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            i =i+1
        else:
            break
    plt.show()

def glcmsearch(imgpath):
    result ={}
    src =[]
    glcmdict ={}
    input = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) # 读取图像，灰度模式 
    for j in range(input.shape[0]):
        for i in range(input.shape[1]):
            input[j][i] = input[j][i] * 32 / 256

    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    glcm = greycomatrix(
        input, [
            2, 8, 16], [
            0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 32, symmetric=True, normed=True)
    #提取特征
    temp =[]
    for prop in ['homogeneity','correlation','contrast']:#对比度，一致性，相关性
        feature=greycoprops(glcm, prop)
        temp.append(np.mean(feature))
        temp.append(np.std(feature,ddof=1))
    entropy = fast_glcm.fast_glcm_entropy(glcm)#熵
    temp.append(np.mean(entropy))
    temp.append(np.std(entropy,ddof=1))
    #print(temp)

    with open("glcm.csv") as f:
        reader = csv.reader(f)
            # loop over the rows in the index
        for row in reader:
            # parse out the imageID and features,
            # then compute the chi-squared distance
            features = [float(x) for x in row[1:]]
            
            glcmdict[row[0]]=np.array(features)
    #print(glcmlist[imgpath])
    for (k,array) in glcmdict.items():
        #d = np.sqrt(np.sum(np.square(temp-array)))
        d = 0.5*np.sum([((a-b)**2)/(a+b+1e-10) for(a,b) in zip(array,temp)])
        result[k] = d
    result = sorted([(v, k) for (k, v) in result.items()])

    i = 1
    plt.figure('glcm')
    testimg = plt.imread(imgpath)
    plt.subplot(3,4,1)
    plt.imshow(testimg)
    plt.axis('off')
    plt.title('testimg')
    for k,v in result:
        if i < 11:
            # print(v)
            img = plt.imread(v)
            plt.subplot(3,4,i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            i =i+1
        else:
            break
    plt.show()

#迭代法求阈值
def diedai(img):
    img_array = np.array(img).astype(np.float32)#转化成数组
    I=img_array
    zmax=np.max(I)
    zmin=np.min(I)
    tk=(zmax+zmin)/2#设置初始阈值
    #根据阈值将图像进行分割为前景和背景，分别求出两者的平均灰度  zo和zb
    b=1
    m,n=I.shape;
    while b==0:
        ifg=0
        ibg=0
        fnum=0
        bnum=0
        for i in range(1,m):
            for j in range(1,n):
                tmp=I(i,j)
                if tmp>=tk:
                    ifg=ifg+1
                    fnum=fnum+int(tmp) #前景像素的个数以及像素值的总和
                else:
                    ibg=ibg+1
                    bnum=bnum+int(tmp)#背景像素的个数以及像素值的总和
        #计算前景和背景的平均值
        zo=int(fnum/ifg)
        zb=int(bnum/ibg)
        if tk==int((zo+zb)/2):
            b=0
        else:
            tk=int((zo+zb)/2)
    return tk

def Husearch(imgpath):
    testarray = []
    res={}
    Hudict = {}
    img = cv2.imread(imgpath)
    #灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值模糊  对椒盐噪声有很好的去燥效果
    grayImage = cv2.medianBlur(grayImage, 5)
    #Sobel算子
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0) #对x求一阶导
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1) #对y求一阶导
    absX = cv2.convertScaleAbs(x)      
    absY = cv2.convertScaleAbs(y)    
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    #二值化
    yvzhi=diedai(Sobel)
    ret,Sobel = cv2.threshold(Sobel, yvzhi, 255, cv2.THRESH_BINARY)
    #hu矩阵
    moments = cv2.moments(Sobel)
    hu_moments = cv2.HuMoments(moments)
    #离心率
    over = measure.regionprops(Sobel,coordinates='xy')#xy消除警告信息
    e = over[0].eccentricity
    
    hu_moments = np.squeeze(hu_moments)#降维
    hu_moments = list(hu_moments)
    hu_moments.append(e)
    hu_moments = np.array(hu_moments)

    with open("Hu.csv") as f:
        reader = csv.reader(f)
            # loop over the rows in the index
        for row in reader:
            features = [float(x) for x in row[1:]]
            Hudict[row[0]]=np.array(features)

    for (k,array) in Hudict.items():
        d = np.sqrt(np.sum(np.square(hu_moments-array)))
        #d = 0.5*np.sum([((a-b)**2)/(a+b+1e-10) for(a,b) in zip(array,hu_moments)])
        res[k] = d
    res = sorted([(v, k) for (k, v) in res.items()])

    i = 1
    plt.figure('Hu')
    testimg = plt.imread(imgpath)
    plt.subplot(3,4,1)
    plt.imshow(testimg)
    plt.axis('off')
    plt.title('testimg')
    for k,v in res:
        if i < 11:
            # print(v)
            img = plt.imread(v)
            plt.subplot(3,4,i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            i =i+1
        else:
            break
    plt.show()


#colorsearch("F:\\CBIR\\tongyi\\accordion\\image_0001.jpg")
#glcmsearch("F:\\CBIR\\tongyi\\ewer\\image_0001.jpg")
#Husearch("F:\\CBIR\\tongyi\\Faces_easy\\image_0001.jpg")
if __name__ == "__main__":
    print("输入图片地址：")
    path = input()
    print("颜色分类输入1,纹理分类输入2,形状分类输入3：")
    n = input()
    if n =='1':
        colorsearch(path)
    elif n=='2':
        glcmsearch(path)
    elif n=='3':
        Husearch(path)
    else:
        print("请输入1,2,3")
