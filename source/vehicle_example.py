#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
from keras.models import model_from_json
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
        help='模型架構檔',
    )
    arg_parser.add_argument(
        '--weights-file',
        required=True,
        help='模型參數檔',
    )
    arg_parser.add_argument(
        '--input-width',
        type=int,
        default=30,
        help='模型輸入影像寬度',
    )
    arg_parser.add_argument(
        '--input-height',
        type=int,
        default=30,
        help='模型輸入影像高度',
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

    pwm1.ChangeDutyCycle(DUTY_CYCLE)
    pwm2.ChangeDutyCycle(DUTY_CYCLE)

    def recognize_image():
        ret, orig_image = video_dev.read()
        assert ret is not None

        # 縮放爲模型輸入的維度、調整數字範圍爲 0～1 之間的數值
        resized_image = cv2.resize(
            orig_image,
            (args.input_width, args.input_height),
        ).astype(np.float32)
        normalized_image = resized_image / 255.0

        batch = normalized_image.reshape(1, args.input_height, args.input_width, 3)
        result_onehot = model.predict(batch)
        class_id = np.argmax(result_onehot, axis=1)[0]

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

    def turn_left():
        pwm1.ChangeDutyCycle(DUTY_CYCLE)
        pwm2.ChangeDutyCycle(0)

    def turn_right():
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(DUTY_CYCLE)

    def stop():
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)

    def right():
        duty = 100

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.5)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(duty)
        time.sleep(0.2)

        pwm1.ChangeDutyCycle(duty)
        pwm2.ChangeDutyCycle(duty)
        time.sleep(0.8)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.5)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(duty)
        time.sleep(0.7)

        pwm1.ChangeDutyCycle(duty)
        pwm2.ChangeDutyCycle(duty)
        time.sleep(0.2)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)

    def left():
        duty = 100

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.5)

        pwm1.ChangeDutyCycle(duty)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.2)

        pwm1.ChangeDutyCycle(duty)
        pwm2.ChangeDutyCycle(duty)
        time.sleep(0.8)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.5)

        pwm1.ChangeDutyCycle(duty)
        pwm2.ChangeDutyCycle(0)
        time.sleep(0.7)

        pwm1.ChangeDutyCycle(duty)
        pwm2.ChangeDutyCycle(duty)
        time.sleep(0.2)

        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)

    def track_line():
        middle_val = GPIO.input(IR_MIDDLE_PIN)
        left_val = GPIO.input(IR_LEFT_PIN)
        right_val = GPIO.input(IR_RIGHT_PIN)

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
            # advice 是 'left', 'right', 'stop', 'other' 之一
            advice = track_line()
            print('advice', advice)

            if advice == 'left':
                turn_left()

            elif advice == 'right':
                turn_right()

            elif advice == 'stop':
                stop()

                sign = recognize_image()

                if sign == 'left':
                    print('left sign')
                    left()

                elif sign == 'right':
                    print('right sign')
                    right()

                elif sign == 'stop':
                    print('stop sign')

                elif sign == 'other':
                    print('no sign')

            elif advice == 'forward':
                forward()

            elif advice == 'stall':
                turn_left()

    except KeyboardInterrupt:
        pass

    # 終止馬達
    pwm1.stop()
    pwm2.stop()


if __name__  == '__main__':
    main()
