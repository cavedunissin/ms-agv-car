# 人工智慧實務工作坊 - 開發智能無人載具範例程式

## 工作流程

**準備工作**

1. 蒐集照片資料，你可以使用我們提供的[資料集](https://drive.google.com/drive/folders/15MIhANzTe6-Tp55Mc-AFZMpaClvDItQv)，課堂上也會介紹建立自己的資料集。
2. 開啓 Azure 工作站，下載本頁提供的範例程式、資料集到工作站上。

**使用 Keras 模型**

1. 使用 `keras_source/train_keras_model.py` 程式建立模型。
2. 下載模型檔到 Raspberry Pi 無人車。
3. 請使用 `keras_source/keras_car.py` 使用我們的模型進行無人自駕。

**使用 TensorFlow + Movidius 模型**

1. 使用 `tf_ncsdk_source/train_tensorflow_model.py` 程式建立模型。
2. 使用 `convert_tf_model.py` 及 `mvNCCompile` 編譯 Movidius 模型。
3. 請使用 `tf_ncsdk_source/movidius_car.py` 使用我們的模型進行無人自駕。

## 相關頁面

- [課程網站](https://www.microsoft.com/taiwan/mstechmrt/mtchandsonlab/ai/08/)
- [課程資源](https://1drv.ms/f/s!AtbELczhnVAca08fYghS639MWFE)
