from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import time
import cv2
CUDA_VISIBLE_DEVICES=0,1,2
np.set_printoptions(threshold=np.inf)
labels = os.listdir('data/train')
labels.insert(0,'無')

def show_img(func,name='show_img'):
    def warp(img):
        img = func(img)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 300, 500)
        cv2.imshow(name, img)
        cv2.waitKey()
        return func   #這段加的話值可以傳出去
    return warp  #這段一定要加

def read_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
    except Exception as e:
        print(img_path,e)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img/255

@show_img
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint16),-1)
    return cv_img

def to_word(pred):
    index = np.argmax(pred)
    return labels[index]


if __name__ == "__main__":
    #請在這邊load 你的model
    model_path = 'InceptionResNetV2.h5'
    model = load_model(model_path)

    for subfolder in os.listdir('0608/'):
        for j in os.listdir('0608/'+subfolder):
            print(j)
            img_path = '0608/' + subfolder + '/' + j

            cv_img = cv_imread(img_path)

            img = read_image(img_path)
            pred = model.predict(img)[0]
            #print(model.predict(img))
            word = to_word(pred)

        print(word)