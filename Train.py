from my_inceptionResNetV2_model import MyInceptionResNetV2_with_CLR

#修改mode
#修改lr
#修改epoch
inceptionResNetV2_model = MyInceptionResNetV2_with_CLR(
    train_dir='data/train',
    test_dir='data/test',
    img_size=(150,150),
    early_stop_patient=15,
    model_name = '801_3.h5',
    epochs=5,
    search_mode=True,
    base_lr=0.00025,
    max_lr=0.003,
    batch_size=64
)
#inceptionResNetV2_model.show_summarys()
inceptionResNetV2_model.train()

