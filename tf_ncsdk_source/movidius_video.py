#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
from mvnc import mvncapi as mvnc


def main():
    # 設定程式參數
    arg_parser = argparse.ArgumentParser(description='使用 Movidius 進行預測')
    arg_parser.add_argument(
        '--graph-file',
        required=True,
        help='Movidius 模型檔',
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

    # 設置 Movidius 裝置
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)
    mvnc_devices = mvnc.enumerate_devices()

    if not mvnc_devices:
        print('找不到 Movidius 裝置')
        exit(1)

    mvnc_dev = mvnc.Device(mvnc_devices[0])
    mvnc_dev.open()

    # 載入模型檔
    try:
        with open(args.graph_file, mode='rb') as file_graph:
            graph_buffer = file_graph.read()
    except (FileNotFoundError, IOError):
        print('無法載入模型檔')
        exit(1)

    graph = mvnc.Graph('graph')
    fifo_in, fifo_out = graph.allocate_with_fifos(mvnc_dev, graph_buffer)

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
            graph.queue_inference_with_fifo_elem(
                fifo_in,
                fifo_out,
                normalized_image,
                None,
            )
            result_onehot, _ = fifo_out.read_elem()

            left_score, right_score, stop_score, other_score = result_onehot
            class_id = np.argmax(result_onehot)

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

    # 終止 Movidius 裝置
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    mvnc_dev.close()
    mvnc_dev.destroy()


if __name__ == '__main__':
    main()
