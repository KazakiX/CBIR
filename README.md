# CBIR
## 摘要
本课题以图像识别感兴趣为目的，构建一个基于内容的图像检索系统（Content Based Image Retrieval, 以下简称 CBIR）。CBIR是计算机视觉领域中关注大规模数字图像内容检索的研究分支。典型的 CBIR 系统，允许用户输入一张图像，在图像数据库（或本地机、或网络）中查找具有相同或相似内容的其它图片。本实训的基本功能要求是实现基于视觉特征的图像检索。
关键词：图像处理，opencv，图像识别，CBIR
## 图片数据集
### 来源
图片数据集在此下载http://www.vision.caltech.edu/Image_Datasets/Caltech101/ 该数据集包括102种共9145张图片。这些图片将全部用于创建CBIR图片数据库。
### 预处理
本次CBIR中，因为各种方法都有不同的预处理方法，只对所有的图像进行了统一大小。所有的图片使用opencv重新定义为200*200的图片。重新定义后的图片存储于“tongyi”文件夹中。相关的代码操作存储在tongyi.py中。
##  CBIR系统构建
### 基于颜色特征识别
在本文构建的CBIR系统中，关于颜色特征的处理选用了RGB空间的直方图作为特征。提取方法使用的python opencv库中的cv2.calcHist()函数。然后使用cv2.normalize()对直方图进行均衡化，提高图像的对比度，以达到图像增强的效果。将提取到的RGB颜色直方图保存在colorhist.csv文件中。在需要时进行读取。

<img src="https://user-images.githubusercontent.com/42568327/114646505-efe79780-9d0d-11eb-8098-89a0e1b48fc7.png" width = "400" heitght="300" alt="颜色识别展示" align=center />

### 基于纹理特征识别
在本CBIR系统中，对于提取图像纹理特征的方法，采用的是灰度共生矩阵的特征方法。将特征值存放在图像特征数据库中。数据保存于glcm.csv文件中。

<img src="https://user-images.githubusercontent.com/42568327/114646515-f4ac4b80-9d0d-11eb-815e-03f4e6912ff0.png" width = "400" heitght="300" alt="颜色识别展示" align=center />

### 基于形状特征识别
作为一组关于形状的统计值，矩不变量的表示形式有多种，如 Hu 矩，具有对图像的旋转、平移和尺度变化的不变性。本CBIR系统的基本思想就是用图像的 Hu 不变矩 u1～u7 和离心率 e 作为图像的形状特征索引，使用适当的相似性距离定义，计算出两幅图像的相似性距离。当距离足够小时，就认为两幅图像是相似的。通过对hu_moments矩阵的一维化，将其转化成list，将e离心率加入，将结果矩阵保存在Hu.csv中。

<img src="https://user-images.githubusercontent.com/42568327/114646520-f6760f00-9d0d-11eb-9ec2-941095e90557.png" width = "400" heitght="300" alt="颜色识别展示" align=center />
