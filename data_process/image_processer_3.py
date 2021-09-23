import cv2
import numpy as np
#import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from scipy import stats
###############轉灰階或二值化###################

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def red1_mask(img):
    #紅1
    lower = np.array([150,80,94])
    upper = np.array([180,255,255])
    redhsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(redhsv1, lower, upper)
    return mask1

def red2_mask(img):
    #紅1
    lower = np.array([0,80,89])
    upper = np.array([10,255,255])
    redhsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(redhsv, lower, upper)
    return mask

def black_mask(img):
    #黑
    lower = np.array([0,0,0])
    upper = np.array([255,255,135])
    redhsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(redhsv, lower, upper)
    return mask

def blue_mask(img):
    #藍
    lower = np.array([90,43,46])
    upper = np.array([124,255,255])
    redhsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(redhsv, lower, upper)
    return mask

def my_dilate(img):
    kernel = np.ones((3, 3), np.uint8)
    new_img = cv2.dilate(img, kernel, iterations=1)
    return new_img

def get_mode(img,w=1.0):
    ##閾值取眾數
    # bincount（）：統計非負整數的個數，不能統計浮點數
    counts = np.bincount(img.flatten())
    #counts的index代表出現的數，counts[index]代表出現數的次數
    #今要求counts[index] 排序後最大跟第二大的counts的index(代表眾數跟出現第二多次的數)
    counts_sort = np.argsort(counts) #最後一個元素是counts最大值的index ，倒數第二是二大
    index = counts_sort[-1]
    #以防圖片出現大量黑色面積
    print('first mode',index)
    if index <= 100: #出現大量黑色區塊的話，取第二多數
        index = counts_sort[-2]
        print('second mode:',index)
        return np.float64(index *w)
    #否則就return原本的眾數
    return index *w

def show_img(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 300, 500)
    cv2.imshow(name, img)

def test_gradient(img):
    #载入灰度原图，并且归一化
    img_original=img
    #分别求X,Y方向的梯度
    grad_X=cv2.Sobel(img_original,-1,2,0,ksize=3)
    grad_Y=cv2.Sobel(img_original,-1,0,2,ksize=3)
    #求梯度图像
    grad=cv2.addWeighted(grad_X,0.5,grad_Y,0.5,0)
    print(grad)
    dst = 255-grad
    show_img('gradient',dst)

def modify_contrast_and_brightness(img,alpha = 4):
    # 公式： Out_img = alpha*(In_img) + beta
    # alpha: alpha參數 (>0)，表示放大的倍数 (通常介於 0.0 ~ 3.0之間)，能夠反應對比度
    # a>1時，影象對比度被放大， 0<a<1時 影象對比度被縮小。
    # beta:  beta参数，用來調節亮度
    # 常數項 beta 用於調節亮度，b>0 時亮度增強，b<0 時亮度降低。

    #眾數
    beta = get_mode(img,w=0.7) *-1
    print(type(beta))
    # add a beta value to every pixel
    img = cv2.add(img, beta)

    # multiply every pixel value by alpha
    img = cv2.multiply(img, alpha)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255)
    return img

def process_img(img,turn=None):
    #紅1
    mask1 = red1_mask(img)
    #紅2
    mask2 = red2_mask(img)
    #紅1 + 紅2 範圍
    mask3 = cv2.bitwise_or(mask1,mask2)
    # 膨脹mask3
    mask3 = my_dilate(mask3)
    #黑白反轉
    mask3 = cv2.bitwise_not(mask3,mask3)
    #show_img('mask3',mask3)
    #灰階
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #取眾數
    image_mode =get_mode(image)
    #把紅色的mask區域換成眾數
    image[mask3==0]= image_mode
    #show_img('final1', image)
    # 高斯後取眾數
    blur = cv2.GaussianBlur(image,(3,3),0)
    #show_img('blur', blur)
    c = get_mode(blur) *0.7

    ret,thresh1 = cv2.threshold(image,c,255,cv2.THRESH_BINARY)
    if turn == 'binary':
        return thresh1
    if turn == 'gray':
        return blur

def train2gray():
    train_dir_name = 'train/'
    for i in os.listdir(train_dir_name):
        subfolder = train_dir_name+i
        for i in os.listdir(subfolder):
            print(subfolder+'/'+i)
            img = cv_imread(subfolder+'/'+i)
            img = process_img(img,turn='gray')
            cv2.imencode('.jpg', img)[1].tofile(subfolder+'/'+i)

def train2binary():
    train_dir_name = 'train/'
    for i in os.listdir(train_dir_name):
        subfolder = train_dir_name+i
        for i in os.listdir(subfolder):
            print(subfolder+'/'+i)
            img = cv_imread(subfolder+'/'+i)
            img = process_img(img,turn='binary')
            cv2.imencode('.jpg', img)[1].tofile(subfolder+'/'+i)

def test2gray():
    train_dir_name = './test/'
    for i in os.listdir(train_dir_name):
        subfolder = train_dir_name+i
        for i in os.listdir(subfolder):
            print(subfolder+'/'+i)
            img = cv_imread(subfolder+'/'+i)
            img = process_img(img,turn='gray')
            cv2.imencode('.jpg', img)[1].tofile(subfolder+'/'+i)

def test2binary():
    train_dir_name = './test/'
    for i in os.listdir(train_dir_name):
        subfolder = train_dir_name+i
        for i in os.listdir(subfolder):
            print(subfolder+'/'+i)
            img = cv_imread(subfolder+'/'+i)
            img = process_img(img,turn='binary')
            cv2.imencode('.jpg', img)[1].tofile(subfolder+'/'+i)

if __name__ == '__main__':

    train2gray()

    # train2binary()
    #
    test2gray()
    #
    # test2binary()


