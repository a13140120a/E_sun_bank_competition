import os
import cv2
import numpy as np
import random
from itertools import product

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def show_img(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 300, 500)
    cv2.waitKey()
    cv2.imshow(name, img)

def random_mask(filename,blocks=0):
    img = cv_imread(filename)
    img_high = img.shape[0]
    img_width = img.shape[1]

    img_high1, img_high2, img_high3, img_high4 = round(img_high*0.2), round(img_high*0.4), round(img_high*0.6), round(img_high*0.8)
    img_high_list = [0,img_high1, img_high2, img_high3, img_high4]

    img_width1, img_width2, img_width3, img_width4 = round(img_width*0.2), round(img_width*0.4), round(img_width*0.6), round(img_width*0.8)
    img_width_list = [0,img_width1, img_width2, img_width3, img_width4]
    #切割成5*5個區塊
    combine = list(product(img_high_list, img_width_list))
    #隨機抽取5個(取後不放回)
    masks = random.sample(combine, blocks)

    mask_high = round(img_high/5)
    mask_widght = round(img_width/5)
    for i in masks:
        #y是寬，x是高
        (x,y) = i
        top_left = (y,x)
        bottom_right_widght = y + mask_widght
        bottom_right_high   = x + mask_high
        if y+img_width*0.2 >img_width :
            bottom_right_widght = img_width
        if x+img_high*0.2 >img_high:
            bottom_right_high = img_high
        bottom_right = (bottom_right_widght, bottom_right_high)
        print('top_left',top_left)
        print('bottom_right',bottom_right)
        cv2.rectangle(img, top_left, bottom_right, 0, -1)
        #img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]= 0
        #show_img('aaa',img)
    return img


if __name__ =="__main__":
    #mask區塊數量
    blocks = 4
    src_dir_name = 'train/'
    target_dir_name = './test/'

    for i in os.listdir(src_dir_name):
        sub_folder = src_dir_name+i+'/'
        for i in os.listdir(sub_folder):
            print(sub_folder+i)

            img1 = random_mask(sub_folder+i,blocks=blocks)
            img2 = random_mask(sub_folder+i,blocks=blocks)
            img3 = random_mask(sub_folder+i,blocks=blocks)
            cv2.imencode('.jpg', img1)[1].tofile(sub_folder+'1_'+i)
            cv2.imencode('.jpg', img2)[1].tofile(sub_folder+'2_'+i)
            cv2.imencode('.jpg', img3)[1].tofile(sub_folder+'3_'+i)
            #show_img('name',img1)

