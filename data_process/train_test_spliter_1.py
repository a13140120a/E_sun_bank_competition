import os
import random
import shutil
import os
import shutil

###分成訓練集跟資料集###


src_dir_name = 'train/'
target_dir_name = './test/'
test_size = 0.3
labels = set(os.listdir(src_dir_name))

def word_classfier():
    word_list_dir = []
    for i in os.listdir(src_dir_name):
        if i.endswith('.jpg'):
            word_list_dir.append(i.split('.')[0][-1])

    word_list_dir = set(word_list_dir)
    print(word_list_dir)

    for i in os.listdir(src_dir_name):
        if i.endswith('.jpg'):
            if i.split('.')[0][-1] in word_list_dir:
                try:
                    os.mkdir(src_dir_name+i.split('.')[0][-1])
                except FileExistsError:
                    pass
                shutil.move(src_dir_name+i,src_dir_name+i.split('.')[0][-1]+'/'+i)


def move_test_data(test_data:list):

    for i in test_data:
        word_subfolder = i.split('.')[0][-1]
        if word_subfolder in labels:
            print(src_dir_name+word_subfolder+'/'+i)
            try:
                os.mkdir(target_dir_name +word_subfolder)
            except FileExistsError:
                pass
            shutil.move(src_dir_name+word_subfolder+'/'+i,target_dir_name +word_subfolder+'/'+i)
        elif  word_subfolder not in labels:
            word_subfolder = i.split('.')[0][0]
            print(src_dir_name+word_subfolder+'/'+i)
            try:
                os.mkdir(target_dir_name +word_subfolder)
            except FileExistsError:
                pass
            try:
                shutil.move(src_dir_name+word_subfolder+'/'+i,target_dir_name +word_subfolder+'/'+i)
            except FileNotFoundError:
                shutil.move(src_dir_name + word_subfolder + '/' + i, target_dir_name + word_subfolder + '/' + i)

def test_train_split():
    try:
        os.mkdir(target_dir_name)
    except FileExistsError:
        pass
    for i in os.listdir(src_dir_name):
        #每個字的照片數
        dir_length = len(os.listdir(src_dir_name+i))
        #3:7抽樣
        test_size = round(0.3 * dir_length)
        test_data = random.sample(os.listdir(src_dir_name+i), k=test_size)
        move_test_data(test_data)

if __name__ == '__main__':
    #把字分類成800個資料夾
    #word_classfier()
    #分成訓練集跟測試集
    test_train_split()