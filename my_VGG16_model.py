# --coding:utf-8--
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Dense,GlobalAvgPool2D
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,BatchNormalization,Dropout,Input, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

from CLR import CyclicLR
from itertools import product
import random
CUDA_VISIBLE_DEVICES=0,1,2


class MyInceptionResNetV2_with_CLR:
    def __init__(self,train_dir, test_dir, img_size, early_stop_patient, model_name, epochs, search_mode:bool, base_lr,max_lr,batch_size):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.epochs = epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.picture_folder = 'pictures'
        self.class_numbers = len(os.listdir(self.train_dir))
        self.batch_size = batch_size


        self.Backbone = VGG16(include_top=False,
                                  weights='imagenet',
                                  input_shape=(self.img_size[0],self.img_size[1],3)
        )
        self.Backbone.trainable = True
        #self.set_backbone_trainable(tops=5)
        self.input_shape1 = Input((self.img_size[0],self.img_size[1],3))
        self.input_shape2 = Input((self.img_size[0],self.img_size[1],3))

        self.encoded_image_1 = VGG16(include_top=False,weights='imagenet',input_shape=(self.img_size[0],self.img_size[1],3))(self.input_shape1)
        self.encoded_image_2 = VGG16(include_top=False,weights='imagenet',input_shape=(self.img_size[0],self.img_size[1],3))(self.input_shape2)
        self.l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([self.encoded_image_1, self.encoded_image_2])
        self.out = Dense(512,activation='relu')(self.l1_distance)
        self.out = Dense(1, activation='sigmoid')(self.out)
        self.model = Model([self.input_shape1, self.input_shape2], self.out)


        self.estop = EarlyStopping(monitor='val_loss', patience=early_stop_patient)
        self.checkpoint = ModelCheckpoint(filepath=model_name,verbose=1,monitor='val_acc', save_best_only=True, mode='auto')
        self.Reduce=ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.2,
                                 patience=5,
                                 verbose=1,
                                 mode='min',
                                 min_delta=0.0001,
                                 cooldown=0,
                                 min_lr=0.0001)

        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.4,
            zoom_range=0.2,
            horizontal_flip=False,
            vertical_flip=False,
            preprocessing_function=self.preprocessing_fun,
            fill_mode='constant',
            cval=255,
            brightness_range=[0.75, 1.5],
        )

        self.test_datagen = ImageDataGenerator(
            rescale=1. / 255,
        )

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical')

        self.test_generator = self.test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)

        self.steps_per_epoch = self.train_generator.samples / self.train_generator.batch_size
        step_size = self.steps_per_epoch * 3
        self.clr_mode= 'triangular2'
        print('samples:',self.train_generator.samples)
        print('batch_size:',self.train_generator.batch_size)
        if search_mode:
            step_size = (self.train_generator.samples * epochs) / self.train_generator.batch_size
            step_size *= 3
            self.clr_mode='triangular'
        print('step_size:', self.steps_per_epoch)
        os.makedirs(self.picture_folder,exist_ok=True)

    def show_summarys(self):
        self.Backbone.summary()
        self.model.summary()

    def set_backbone_trainable(self,tops = 5):
        for layer in self.Backbone.layers[:-tops]:
           layer.trainable = False
        for layer in self.Backbone.layers[-tops:]:
           layer.trainable = True

    def train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'] )
        # ????????????????????? ????????????
        H = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,  # ??????????????????
            validation_data=self.test_generator,
            validation_steps=64,
            callbacks=[self.checkpoint, self.estop, self.Reduce]
        )
        self.show_total_history_picture(H)

    def random_mask(self,img, blocks=0):

        img_high = img.shape[0]
        img_width = img.shape[1]

        img_high1, img_high2, img_high3, img_high4 = round(img_high * 0.2), round(img_high * 0.4), round(
            img_high * 0.6), round(img_high * 0.8)
        img_high_list = [0, img_high1, img_high2, img_high3, img_high4]

        img_width1, img_width2, img_width3, img_width4 = round(img_width * 0.2), round(img_width * 0.4), round(
            img_width * 0.6), round(img_width * 0.8)
        img_width_list = [0, img_width1, img_width2, img_width3, img_width4]
        # ?????????5*5?????????
        combine = list(product(img_high_list, img_width_list))
        # ????????????5???(???????????????)
        masks = random.sample(combine, blocks)

        mask_high = round(img_high / 5)
        mask_widght = round(img_width / 5)
        for i in masks:
            # y????????????x?????????
            (x, y) = i
            # ??????????????????????????????????????????????????????????????????high, width ??????
            top_left = (y, x)
            bottom_right_widght = y + mask_widght
            bottom_right_high = x + mask_high
            if y + img_width * 0.2 > img_width:
                bottom_right_widght = img_width
            if x + img_high * 0.2 > img_high:
                bottom_right_high = img_high
            bottom_right = (bottom_right_widght, bottom_right_high)

            # cv2.rectangle(img, top_left, bottom_right, 0, -1)
            img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 0
        return img

    def preprocessing_fun(self,img):

        # ??????
        # new_img = np.array(img,dtype=np.uint8)
        # new_img = random_mask(new_img,blocks=3)
        # cv2.imshow('aa',new_img)
        # cv2.waitKey(0)
        # return img

        # ??????
        new_img = self.random_mask(img, blocks=5)

        return new_img

    def show_total_history_picture(self,H):
        accuracy = H.history['acc']
        #epoch
        val_accuracy = H.history['val_acc']
        plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training train)')
        plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation train)')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig(self.picture_folder+'/acc.png')
        plt.show()

        # ??????loss????????????
        loss = H.history['loss']
        val_loss = H.history['val_loss']
        plt.plot(range(len(loss)), loss, marker='.', label='loss(training train)')
        plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(evaluation train)')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.picture_folder+'/loss.png')
        plt.show()

