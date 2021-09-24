# E_sun_bank_competition

1. 資料前處理
    * [train_test_spliter_1.py](https://github.com/a13140120a/E_sun_bank_competition/blob/master/data_process/train_test_spliter_1.py) 把資料分成訓練集跟測試集  
    ```py
   test_size = 0.3  # 修改這邊可以調整測試集大小
    ```
    * [Data_expander_2.py](https://github.com/a13140120a/E_sun_bank_competition/blob/master/data_process/Data_expander_2.py) 進行隨機mask 部分區塊的資料擴增  
      效果圖:  
      ![image](https://github.com/a13140120a/E_sun_bank_competition/blob/master/imgs/mask_result.PNG)
2. 訓練模型
   * [my_inceptionResNetV2_model.py](https://github.com/a13140120a/E_sun_bank_competition/blob/master/my_inceptionResNetV2_model.py) Backbone 使用的是InceptionResnetV2 的架構，top 的部份加上一個GlobalAvgPool2D 層跟兩個FC。
   * [CLR.py](https://github.com/a13140120a/E_sun_bank_competition/blob/master/CLR.py) 優化器使用CLR ，有興趣的人可以查閱以下連結: [https://github.com/bckenstler/CLR](https://github.com/bckenstler/CLR)
   
3. 評估模型
   * [acc_test.py](https://github.com/a13140120a/E_sun_bank_competition/blob/master/acc_test.py) 評估模型使用的是keras 的evaluate_generator，並且可以使用[check.py](https://github.com/a13140120a/E_sun_bank_competition/blob/master/check.py) 查看模型的lr 與對應的acc 進一步做超參數調整