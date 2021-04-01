import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import measure,color

import cv2

path = 'F:\\CBIR\\tongyi'

#迭代法
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


for root,dirs,files in os.walk(path):
    for name in tqdm(files):
        imgpath = os.path.join(root,name)
        #读取图像
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
        #迭代法阈值二值化
        yvzhi=diedai(Sobel)
        ret,Sobel = cv2.threshold(Sobel, yvzhi, 255, cv2.THRESH_BINARY)
        #hu矩阵
        moments = cv2.moments(Sobel)
        hu_moments = cv2.HuMoments(moments)

        #离心率
        over = measure.regionprops(Sobel,coordinates='xy')
        e = over[0].eccentricity
        output = open("Hu.csv","a+")
        features = [str(f[0]) for f in hu_moments]
        output.write("%s,%s,%s\n" %(imgpath,",".join(features),str(e)))
        output.close()
