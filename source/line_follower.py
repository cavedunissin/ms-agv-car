#!/usr/bin/env python3
import time

import RPi.GPIO as GPIO

# 設定腳位
PWM_PIN_left = 17
PWM_PIN_right = 18

IR_LEFT_PIN = 2
IR_MIDDLE_PIN = 3
IR_RIGHT_PIN = 4

DUTY_CYCLE = 80


def main():
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
            # advice 是 'left', 'right', 'stop', 'other' 之一
            advice = track_line()

            if advice == 'left':
                print('動作:', '左轉')
                turn_left()

            elif advice == 'right':
                print('動作:', '右轉')
                turn_right()

            elif advice == 'stop':
                print('動作:', '停止')
                stop()

            elif advice == 'forward':
                print('動作:', '前進')
                forward()

            elif advice == 'stall':
                print('動作:', '左轉')
                turn_left()

    except KeyboardInterrupt:
        print('使用者中斷')

    # 終止馬達
    pwm1.stop()
    pwm2.stop()


if __name__  == '__main__':
    main()
