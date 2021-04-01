import numpy as np
import os
import cv2
import time
from tqdm import tqdm
from PIL import Image
path = 'F:\\CBIR\\101_ObjectCategories'

for root,dirs,files in tqdm(os.walk(path)):
    # for name in dirs:
    #     path = 'F:\\CBIR\\tongyi\\'+name
    #     os.mkdir(path)
    for name in files:
        imgpath = os.path.join(root,name)
        img_array=cv2.imread(imgpath,cv2.IMREAD_COLOR)
        #'''调用cv2.resize函数resize图片'''
        new_array=cv2.resize(img_array,(200,200))
        #'''生成图片存储的目标路径'''
        name = imgpath.split('\\')[-2]
        iname =imgpath.split('\\')[-1]
        iname = iname.split('.')[0]
        save_path='F:\\CBIR\\tongyi\\'+name+'\\'+iname+'.jpg'
        #'''调用cv.2的imwrite函数保存图片'''
        cv2.imwrite(save_path,new_array)
        print(save_path)
# path='F:\\CBIR\\101_ObjectCategories\\accordion\\image_0002.jpg'
# name = path.split('\\')[-2]