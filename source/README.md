# 範例程式

## train\_model.py

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

指令詳細說明請參見 `-h` 選項，訓練過的模型可以參考上層的 `keras_model` 目錄。

```sh
# 使用自定模型，模型定義見原始碼
# 訓練 64 回合，完成後儲存模型檔於 model.json、儲存模型參數於 weight.h5
# 將預測結果輸出在 output.txt
python3 ./train_model.py --model-file model.json \
                         --weights-file weight.h5 \
                         --data-dir /path/to/dataset
```

## keras\_video.py

本範例載入 `train_model.py` 訓練後儲存的模型檔及模型參數檔，讀取影像進行辨識。

```sh
# 辨識攝影機
python3 ./keras_video.py --model-file model.json \
                         --weights-file weights.h5
```

## data\_collect.py

自動照片蒐集程式，令軌跡車循跡前進同時拍攝畫面。

```sh
# 儲存到 my_data/ 目錄
python3 ./data_collect.py --data-dir my_data/
```

## vehicle\_example.py

軌跡車主程式，沿着軌跡移動的同時，使用訓練好的 Keras 模型辨識路牌。

```sh
# model.json、weights.h5 是 train_model.py 訓練生成的模型
python3 ./vehicle_example.py --model-file model.json \
                             --weights-file weights.h5
```

## line\_follower.py

軌跡車範例程式，使用光感應器循軌跡行走。

```sh
./line_follower.py
```
