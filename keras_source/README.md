## 使用 Keras 建立路牌分類模型

本範例使用 Keras 訓練模型，資料目錄請依照下面方式編排。訓練用圖片分類成 `left`、`right`、`stop` 目錄，非上述三類放置在 `other`目錄。測試資料置於 `test` 目錄。

```
data_dir
├── left
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
├── right
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
├── stop
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
├── other
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
└── test
    ├── 000.jpg
    ├── 001.jpg
    └── ...
```

執行 `train_keras_model.py` 指令訓練模型，詳細說明請參見 `-h` 選項。

上層的 `models/keras_model` 目錄也有訓練好的模型可參考。

```sh
# 完成後儲存模型檔於 model.json、儲存模型參數於 weight.h5
./train_keras_model.py --model-file model.json \
                 --weights-file weight.h5 \
                 --data-dir ~/dataset
```

本範例載入 `train_keras_model.py` 訓練後儲存的模型檔及模型參數檔，讀取影像進行辨識。

```sh
# 辨識攝影機範例
./keras_video.py --model-file model.json \
                 --weights-file weights.h5

# 辨識影片檔範例，並指定輸入模型的維度
./keras_video.py --video-type file \
                 --source ../sample_video/example_1.mp4 \
                 --model-file model.json \
                 --weights-file weights.h5
```

`keras_car.py` 是結合軌跡車和 Keras 模型的範例。

```sh
./keras_car.py --model-file model.json \
               --weights-file weights.h5
```
