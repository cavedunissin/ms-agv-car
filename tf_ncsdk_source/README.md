## 使用 TensorFlow + Movidius 進行路牌分類

本範例使用 TensorFlow 訓練模型，資料目錄請依照下面方式編排。訓練用圖片分類成 `left`、`right`、`stop` 目錄，非上述三類放置在 `other`目錄。測試資料置於 `test` 目錄。

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

執行 `train_tensorflow_model.py` 訓練模型，指令詳細說明請參見 `-h` 選項。

上層的 `models/tf_ncsdk_model` 目錄也有訓練好的模型可參考。

```sh
# 訓練 128 回合，完成輸出模型到 saved_model 目錄
./train_tensorflow_model.py --model-base-dir saved_model \
                            --data-dir ~/dataset
```

將模型編譯為 Movidius 模型檔，需要兩個步驟，先執行 `convert_tf_model.py` 將模型轉換為 ncsdk 可讀取的格式，再執行 `mvNCCompile` 編譯為 Movidius 模型檔。

```sh
# 路徑中的 XXXXXXXXXX 請依據實際路徑填寫，指令完成時會輸出 converted_mode 目錄
./convert_tf_model.py --saved-model-dir saved_model/XXXXXXXXXX \
                      --output-model-dir converted_model

# 轉換為 ncsdk 格式模型檔 model.graph
mvNCCompile converted_model/model.meta \
            -s 12 \
            -in input_image
            -on probabilities \
            -o model.graph
```

講模型檔 model.graph 移到 RPi 上，在上面執行 `movidius_video.py` 使用攝影機測試模型。

```sh
./movidius_video.py --graph-file model.graph \
                    --video-type camera \
                    --source /dev/video0
```

`movidius_car.py` 是結合軌跡車和 Movidius 模型的範例。

```sh
./movidius_car.py --model-file model.graph
```
