#!/usr/bin/env python3
import os
import time
import argparse

import RPi.GPIO as GPIO
import cv2

# 設定腳位
PWM_PIN_left = 17
PWM_PIN_right = 18

IR_LEFT_PIN= 2
IR_MIDDLE_PIN= 3
IR_RIGHT_PIN=  4

DUTY_CYCLE = 70
IMAGE_QUEUE_LIMIT = 400

def main():
    # 設定程式參數
    arg_parser = argparse.ArgumentParser(description='軌跡車程式。')
    arg_parser.add_argument('--data-dir', required=True)

    # 解讀程式參數
    args = arg_parser.parse_args()

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

    images = list()

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

    def track_line():
        # TODO verify
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
            # 根據光感應器讀值決定動作
            # advice 是 'left', 'right', 'stop', 'other' 之一
            advice = track_line()
            print('advice', advice)

            if advice == 'left':
                turn_left()

            elif advice == 'right':
                turn_right()

            elif advice == 'stop':
                stop()

            elif advice == 'forward':
                forward()

            elif advice == 'stall':
                turn_left()

            # 拍攝照片並儲存到序列
            ret, image = video_dev.read()
            images.append((os.path.join(args.data_dir, '%d-%s.jpg' % (time.time(), advice)), image))

            print('queue size: %d' % (len(images) + 1))

            # 若序列大小到達限制，停下車，並將序列的照片存入硬碟
            if len(images) == IMAGE_QUEUE_LIMIT:
                stop()

                # 儲存影像
                for path, image in images:
                    print('Write %s' % path)
                    cv2.imwrite(path, image)

                del images
                images = list()


    except KeyboardInterrupt:
        pass

    # 終止馬達
    pwm1.stop()
    pwm2.stop()

    # 將序列的照片存入硬碟
    for path, image in images:
        print('Write %s' % path)
        cv2.imwrite(path, image)


if __name__  == '__main__':
    main()
