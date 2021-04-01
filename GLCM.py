import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import fast_glcm
from tqdm import tqdm
temp = []
def get_inputs(s): # s为图像路径
    input = cv2.imread(s, cv2.IMREAD_GRAYSCALE) # 读取图像，灰度模式 
    for j in range(input.shape[0]):
        for i in range(input.shape[1]):
            input[j][i] = input[j][i] * 32 / 256

    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    glcm = greycomatrix(
        input, [
            2, 8, 16], [
            0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 32, symmetric=True, normed=True)
            
    print(glcm) 
    
    #得到共生矩阵统计值，http://tonysyu.github.io/scikit-image/api/skimage.feature.html#skimage.feature.greycoprops
    for prop in {'contrast', 'dissimilarity',
                 'homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = greycoprops(glcm, prop)
        # temp=np.array(temp).reshape(-1)
        print(prop, temp)
        print(np.mean(temp),np.std(temp,ddof=1))
        glcm_ent = fast_glcm.fast_glcm_entropy(glcm)
        print(glcm_ent)
    # plt.imshow(input,cmap="gray")
    # plt.show()


path = "F:\\CBIR\\tongyi"

for root,dirs,files in os.walk(path):
    for name in tqdm(files):
        imgpath = os.path.join(root,name)
        input = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) # 读取图像，灰度模式 
        for j in range(input.shape[0]):
            for i in range(input.shape[1]):
                input[j][i] = input[j][i] * 32 / 256

        glcm = greycomatrix(
            input, [
                2, 8, 16], [
                0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 32, symmetric=True, normed=True)
        
        for prop in {'contrast', 'homogeneity', 'correlation'}:#对比度，一致性，相关性
            feature=greycoprops(glcm, prop)
            temp.append(np.mean(feature))
            temp.append(np.std(feature,ddof=1))

        entropy = fast_glcm.fast_glcm_entropy(glcm)#熵
        temp.append(np.mean(entropy))
        temp.append(np.std(entropy,ddof=1))
        
        output = open("glcm.csv","a+")
        features = [str(f) for f in temp]
        output.write("%s,%s\n" %(imgpath,",".join(features)))
        output.close()

        temp.clear()

