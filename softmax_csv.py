from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model
import time
import csv

def read_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
    except Exception as e:
        print(img_path,e)
        return
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img/255

def to_word(pred:np.array)->str:
    index = np.argmax(pred)
    return labels[index]

def append_pre_true(arr,pre,true,predict_true_or_not)->list:
    arr = list(arr)
    arr.append(pre)
    arr.append(true)
    arr.append(predict_true_or_not)
    return arr

def predict_true_or_not(pre:str, true:str)->int:
    if pre == true:
        return 1
    else:
        return 0

def columns(labels:list):

    for i in ['pre', 'label', 'true_or_not']:
        labels.append(i)
    return labels

def model_predict(model,img):
    pred = model.predict(img)[0]
    word = to_word(pred)
    true_or_not = predict_true_or_not(word, subfolder)
    return pred, word, true_or_not


if __name__ == "__main__":
    #800字標籤
    labels = os.listdir('data/train')
    #labels.insert(0,'無')
    #請在這邊load 你的model
    model_path = './800_compare.h5'
    model = load_model(model_path)
    #存放檔案的資料夾
    folder_name = './official_in_800/'
    #csv檔名稱
    csv_name = "official_in_800.csv"

    with open(csv_name, "w",newline="",encoding="utf_8_sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns(labels))  #先寫入欄位名稱
        for subfolder in os.listdir(folder_name):
            for jpg in os.listdir(folder_name+subfolder):
                # 請輸入你的圖片path
                img_path = folder_name+subfolder+'/'+jpg
                img = read_image(img_path)
                #1*800向量，預測字，是否猜中
                pred, word, true_or_not = model_predict(model,img)
                #變成一列 : 1*800向量，預測字，真實字，是否猜中
                row = append_pre_true(pred, word, subfolder, true_or_not)
                writer.writerow(row)
                print(word,subfolder,true_or_not)

