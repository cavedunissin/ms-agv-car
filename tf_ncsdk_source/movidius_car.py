#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
from mvnc import mvncapi as mvnc
import RPi.GPIO as GPIO

# 設定腳位
PWM_PIN_left = 17
PWM_PIN_right = 18

IR_LEFT_PIN = 2
IR_MIDDLE_PIN = 3
IR_RIGHT_PIN = 4

DUTY_CYCLE = 80


def main():
    # 設定程式參數
    arg_parser = argparse.ArgumentParser(description='軌跡車程式。')
    arg_parser.add_argument(
        '--model-file',
        required=True,
        help='Movidius 模型檔',
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
        with open(args.model_file, mode='rb') as file_graph:
            graph_buffer = file_graph.read()
    except (FileNotFoundError, IOError):
        print('無法載入模型檔')
        exit(1)

    graph = mvnc.Graph('graph')
    fifo_in, fifo_out = graph.allocate_with_fifos(mvnc_dev, graph_buffer)

    # 開啓影片來源
    video_dev = cv2.VideoCapture(0)

    # 初始化 GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PWM_PIN_left, GPIO.OUT)
    GPIO.setup(PWM_PIN_right, GPIO.OUT)
    pwm1 = GPIO.PWM(PWM_PIN_left, 500)
    pwm2 = GPIO.PWM(PWM_PIN_right, 500)
    pwm1.start(0)
    pwm2.start(0)

    GPIO.setup(IR_RIGHT_PIN, GPIO.IN)  #GPIO 2 -> Left IR out
    GPIO.setup(IR_MIDDLE_PIN, GPIO.IN) #GPIO 3 -> Right IR out
    GPIO.setup(IR_LEFT_PIN, GPIO.IN)   #GPIO 4 -> Right IR out

    def recognize_image():
        ret, orig_image = video_dev.read()
        assert ret is not None

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
        class_id = np.argmax(result_onehot)
        left_score, right_score, stop_score, other_score = result_onehot

        print('預測：%.2f %.2f %.2f %.2f' % (left_score, right_score, stop_score, other_score))

        # print(result_onehot)
        if class_id == 0:
            return 'left'
        elif class_id == 1:
            return 'right'
        elif class_id == 2:
            return 'stop'
        elif class_id == 3:
            return 'other'

    def forward():
        pwm1.ChangeDutyCycle(DUTY_CYCLE)
        pwm2.ChangeDutyCycle(DUTY_CYCLE)

    def head_left():
        pwm1.ChangeDutyCycle(DUTY_CYCLE)
        pwm2.ChangeDutyCycle(0)

    def head_right():
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(DUTY_CYCLE)

    def stop():
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)

    def cross_left():
        time.sleep(1)

        pwm1.ChangeDutyCycle(100)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.35)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        time.sleep(1)

        pwm1.ChangeDutyCycle(100)
        pwm2.ChangeDutyCycle(100)
        time.sleep(1)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.5)

    def cross_right():
        time.sleep(1)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(100)
        time.sleep(0.35)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        time.sleep(1)

        pwm1.ChangeDutyCycle(100)
        pwm2.ChangeDutyCycle(100)
        time.sleep(1)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.5)

    def track_line():
        middle_val = GPIO.input(IR_MIDDLE_PIN)
        left_val = GPIO.input(IR_LEFT_PIN)
        right_val = GPIO.input(IR_RIGHT_PIN)
        print('光感:', left_val, middle_val, right_val)

        if middle_val:
            if left_val and right_val:        # 白白白
                return 'stop'
            elif left_val and not right_val:  # 白白黑
                return 'left'
            elif not left_val and right_val:  # 黑白白
                return 'right'
            else:
                return 'forward'              # 黑白黑
        else:
            if left_val and right_val:        # 白黑白
                return 'stall'
            elif left_val and not right_val:  # 白黑黑
                return 'left'
            elif not left_val and right_val:  # 黑黑白
                return 'right'
            else:                             # 黑黑黑
                return 'stall'

    try:
        while True:
            advice = track_line()

            if advice == 'left':
                print('動作:', '左轉')
                head_left()

            elif advice == 'right':
                print('動作:', '右轉')
                head_right()

            elif advice == 'stop':
                print('動作:', '停止')
                stop()

                sign = recognize_image()

                if sign == 'left':
                    print('影像:', '左轉標誌')
                    cross_left()

                elif sign == 'right':
                    print('影像:', '右轉標誌')
                    cross_right()

                elif sign == 'stop':
                    print('影像:', '停止標誌')

                elif sign == 'other':
                    print('影像:', '無標誌')

            elif advice == 'forward':
                print('動作:', '前進')
                forward()

            elif advice == 'stall':
                print('動作:', '左轉')
                head_left()

            print()

    except KeyboardInterrupt:
        print('使用者中斷')

    # 終止馬達
    pwm1.stop()
    pwm2.stop()

    # 終止影像裝置
    video_dev.release()

    # 終止 Movidius 裝置
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    mvnc_dev.close()
    mvnc_dev.destroy()


if __name__  == '__main__':
    main()
