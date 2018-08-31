#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
from keras.models import model_from_json


def main():
    # 設定程式參數
    arg_parser = argparse.ArgumentParser(description='執行 Keras 模型，辨識影片檔或攝影機影像。')
    arg_parser.add_argument(
        '--model-file',
        required=True,
        help='模型架構檔',
    )
    arg_parser.add_argument(
        '--weights-file',
        required=True,
        help='模型參數檔',
    )
    arg_parser.add_argument(
        '--video-type',
        choices=['file', 'camera'],
        default='camera',
        help='影片類型',
    )
    arg_parser.add_argument(
        '--source',
        default='/dev/video0',
        help='影片來源檔',
    )
    arg_parser.add_argument(
        '--input-width',
        type=int,
        default=48,
        help='模型輸入影像寬度',
    )
    arg_parser.add_argument(
        '--input-height',
        type=int,
        default=48,
        help='模型輸入影像高度',
    )
    arg_parser.add_argument(
        '--gui',
        action='store_true',
        help='啓用圖像界面',
    )

    # 解讀程式參數
    args = arg_parser.parse_args()
    assert args.input_width > 0 and args.input_height > 0

    # 載入模型
    with open(args.model_file, 'r') as file_model:
        model_desc = file_model.read()
        model = model_from_json(model_desc)

    model.load_weights(args.weights_file)

    # 開啓影片來源
    if args.video_type == 'file':  # 檔案
        video_dev = cv2.VideoCapture(args.source)
        video_width = video_dev.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT)

    elif args.video_type == 'camera':  # 攝影機
        video_dev = cv2.VideoCapture(0)

    # 主迴圈
    try:
        prev_timestamp = time.time()

        while True:
            ret, orig_image = video_dev.read()
            curr_time = time.localtime()

            # 檢查串流是否結束
            if ret is None or orig_image is None:
                break

            # 縮放爲模型輸入的維度、調整數字範圍爲 0～1 之間的數值
            resized_image = cv2.resize(
                orig_image,
                (args.input_width, args.input_height),
            ).astype(np.float32)
            normalized_image = resized_image / 255.0

            # 執行預測
            batch = normalized_image.reshape(1, args.input_height, args.input_width, 3)
            result_onehot = model.predict(batch)
            left_score, right_score, stop_score, other_score = result_onehot[0]
            class_id = np.argmax(result_onehot, axis=1)[0]

            if class_id == 0:
                class_str = 'left'
            elif class_id == 1:
                class_str = 'right'
            elif class_id == 2:
                class_str = 'stop'
            elif class_id == 3:
                class_str = 'other'

            # 計算執行時間
            recent_timestamp = time.time()
            period = recent_timestamp - prev_timestamp
            prev_timestamp = recent_timestamp

            print('時間：%02d:%02d:%02d ' % (curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec))
            print('輸出：%.2f %.2f %.2f %.2f' % (left_score, right_score, stop_score, other_score))
            print('類別：%s' % class_str)
            print('費時：%f' % period)
            print()

            # 顯示圖片
            if args.gui:
                cv2.imshow('', orig_image)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print('使用者中斷')

    # 終止影像裝置
    video_dev.release()


if __name__ == '__main__':
    main()
