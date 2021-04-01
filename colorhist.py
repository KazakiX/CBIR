import csv
import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2

path = 'F:\\CBIR\\tongyi'

for root,dirs,files in tqdm(os.walk(path)):
    for name in files:
        imgpath = os.path.join(root,name)
        #print(imgpath)
        imgCV = cv2.imread(imgpath)
        #提取直方图
        Hist = cv2.calcHist([imgCV],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
        Hist = cv2.normalize(Hist,Hist,0,255,cv2.NORM_MINMAX).flatten()
        
        output = open("color.csv","a+")
        features = [str(f) for f in Hist]
        output.write("%s,%s\n" %(imgpath,",".join(features)))
        output.close()
