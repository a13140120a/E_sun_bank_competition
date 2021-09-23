from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.models import load_model

labels = os.listdir('data/train')
c = 0

# 請在這邊load 你的model
model_path = 'InceptionResNetV2_checkpoint.h5'
model = load_model(model_path)

folder_name = './unofficial_in_800/'
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    folder_name,
    target_size=(150, 150),
    batch_size=50,
    class_mode='categorical',
    shuffle=False)

test_generator.reset()
test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples // 50,
                                               verbose=1)
print('test acc:', test_acc)
print('test loss:', test_loss)