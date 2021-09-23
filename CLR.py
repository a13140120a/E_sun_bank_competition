import os

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from itertools import product
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import cv2


class CyclicLR(Callback):
    """

    非官方+官方800 :
            base_lr=0.0005,
            max_lr=0.003,
            strp_size = steps_per_epoch*3
    """

    def __init__(
            self,
            base_lr=0.00025,
            max_lr=0.003,
            step_size=2000.,
            mode='triangular2',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle',
            picture_folder='pictures'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.analyze_data_dir_name = 'analyze/'
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.picture_folder = picture_folder
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self.epoch_logs = {}

        os.makedirs(self.analyze_data_dir_name+'batch/', exist_ok=True)
        os.makedirs(self.analyze_data_dir_name+'epoch/', exist_ok=True)
        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())
        print(K.get_value(self.model.optimizer.lr))

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        # print(self.history)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        #
        for k, v in logs.items():
            self.epoch_logs.setdefault(k, []).append(v)

        print(logs)

    def on_train_end(self, logs=None):
        self.plot_lr()
        self.plot_acc()
        self.plot_acc_smooth()
        self.save_train_data()

    def save_train_data(self):
        #每個batch資料
        with open(self.analyze_data_dir_name+ 'batch/' +'/lr.pickle','wb') as file:
            lr = self.history['lr']
            pickle.dump(lr,file)
        with open(self.analyze_data_dir_name + 'batch/' + '/acc.pickle','wb') as file:
            acc = self.history['acc']
            pickle.dump(acc,file)
        #每個epoch資料
        with open(self.analyze_data_dir_name + 'epoch/' + '/loss.pickle','wb') as file:
            epoch_loss = self.epoch_logs['loss']
            pickle.dump(epoch_loss,file)
        with open(self.analyze_data_dir_name + 'epoch/' + '/acc.pickle','wb') as file:
            epoch_acc = self.epoch_logs['acc']
            pickle.dump(epoch_acc,file)
        with open(self.analyze_data_dir_name + 'epoch/' + '/val_loss.pickle','wb') as file:
            epoch_val_loss = self.epoch_logs['val_loss']
            pickle.dump(epoch_val_loss,file)
        with open(self.analyze_data_dir_name + 'epoch/' + '/val_acc.pickle','wb') as file:
            epoch_val_acc = self.epoch_logs['val_acc']
            pickle.dump(epoch_val_acc,file)
        with open(self.analyze_data_dir_name + 'epoch/' + '/lr.pickle','wb') as file:
            epoch_lr = self.epoch_logs['lr']
            pickle.dump(epoch_lr,file)


    def plot_lr(self):
        # Helper function to quickly inspect the learning rate schedule.
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.savefig(self.picture_folder+"/plot_lr.png")
        plt.show()

    def plot_acc(self):
        # '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['acc'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy')
        plt.savefig(self.picture_folder+"/plot_acc.png")
        plt.show()

    def plot_acc_smooth(self,smooth=10):
        '''

        :param smooth:  把圖分成幾個點(會比較平滑)
        :return: None
        '''
        gap = round(self.step_size/smooth)
        lr = []
        acc = []
        for i in range(0,len(self.history['lr']),gap):
            lr.append(self.history['lr'][i])
        for i in range(0,len(self.history['acc']),gap):
            acc.append(self.history['acc'][i])

        plt.plot(lr, acc)
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy')
        plt.savefig("./plot_acc_mini.png")
        plt.show()


def random_mask(img,blocks=0):

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
        #y是橫軸，x是縱軸
        (x,y) = i
        #第一個是橫軸，第二個是縱軸，所以這邊跟上面的high, width 相反
        top_left = (y,x)
        bottom_right_widght = y + mask_widght
        bottom_right_high   = x + mask_high
        if y+img_width*0.2 >img_width :
            bottom_right_widght = img_width
        if x+img_high*0.2 >img_high:
            bottom_right_high = img_high
        bottom_right = (bottom_right_widght, bottom_right_high)

        #cv2.rectangle(img, top_left, bottom_right, 0, -1)
        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 0
    return img

def preprocessing_fun(img):

    #查看
    # new_img = np.array(img,dtype=np.uint8)
    # new_img = random_mask(new_img,blocks=3)
    # cv2.imshow('aa',new_img)
    # cv2.waitKey(0)
    # return img

    #執行
    new_img = random_mask(img,blocks=5)

    return new_img
