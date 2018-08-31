#!/usr/bin/env python3
import os
import glob
import argparse
import logging

import cv2
import numpy as np
import tensorflow as tf

X_FEATURE_KEY = 'x'


def load_data(data_dir, input_height, input_width, input_channel, n_classes):
    # 搜尋所有圖檔
    match_left = os.path.join(data_dir, 'left', '*.jpg')
    paths_left = glob.glob(match_left)

    match_right = os.path.join(data_dir, 'right', '*.jpg')
    paths_right = glob.glob(match_right)

    match_stop = os.path.join(data_dir, 'stop', '*.jpg')
    paths_stop = glob.glob(match_stop)

    match_other = os.path.join(data_dir, 'other', '*.jpg')
    paths_other = glob.glob(match_other)

    match_test = os.path.join(data_dir, 'test', '*.jpg')
    paths_test = glob.glob(match_test)

    n_train = len(paths_left) + len(paths_right) + len(paths_stop) + len(paths_other)
    n_test = len(paths_test)

    # 初始化資料集矩陣
    train_x = np.zeros(
        shape=(n_train, input_height, input_width, input_channel),
        dtype='float32',
    )
    train_y = np.zeros(
        shape=(n_train,),
        dtype='int32',
    )
    test_x = np.zeros(
        shape=(n_test, input_height, input_width, input_channel),
        dtype='float32',
    )

    # 讀取圖片到資料集
    paths_train = paths_left + paths_right + paths_stop + paths_other

    for ind, path in enumerate(paths_train):
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (input_width, input_height))
        train_x[ind] = resized_image

    for ind, path in enumerate(paths_test):
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (input_width, input_height))
        test_x[ind] = resized_image

    # 設定訓練集的標記
    n_left = len(paths_left)
    n_right = len(paths_right)
    n_stop = len(paths_stop)
    n_other = len(paths_other)

    begin_ind = 0
    end_ind = n_left
    train_y[begin_ind:end_ind] = 0

    begin_ind = n_left
    end_ind = n_left + n_right
    train_y[begin_ind:end_ind] = 1

    begin_ind = n_left + n_right
    end_ind = n_left + n_right + n_stop
    train_y[begin_ind:end_ind] = 2

    begin_ind = n_left + n_right + n_stop
    end_ind = n_left + n_right + n_stop + n_other
    train_y[begin_ind:end_ind] = 3

    # 正規化數值到 0～1 之間
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    return (paths_train, train_x, train_y), (paths_test, test_x,)


def custom_model_fn(features, labels, mode, params):
    training = bool(mode == tf.estimator.ModeKeys.TRAIN)

    # 區塊函數，由多層 conv2d、batch_normalization 構成
    def conv_block(x, filters):
        # Movidius compiler does not support FusedBatchNorm operator.
        # To avoid this error, pass fused=False to batch_normalization()

        # x = tf.layers.batch_normalization(
        #     x,
        #     training=training,
        #     fused=False,
        # )
        x = tf.layers.conv2d(
            x,
            filters=filters,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
        )

        # x = tf.layers.batch_normalization(
        #     x,
        #     training=training,
        #     fused=False,
        # )
        shortcut = x
        x = tf.layers.conv2d(
            x,
            filters=filters,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
        )
        x = tf.add(x, shortcut)
        x = tf.layers.max_pooling2d(
            x,
            pool_size=(2, 2),
            strides=(2, 2),
        )

        return x

    # 輸入層
    input_image = features[X_FEATURE_KEY]

    # 重複輸入區塊，每次用不同數量的 filter
    x = conv_block(input_image, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    # 最終區塊
    x = tf.layers.flatten(x)
    # x = tf.layers.batch_normalization(
    #     x,
    #     training=training,
    #     fused=False,
    # )
    x = tf.layers.dense(
        x,
        units=512,
        activation=tf.nn.relu,
    )
    x = tf.layers.dense(
        x,
        units=512,
        activation=tf.nn.relu,
    )

    # 模型輸出
    logits = tf.layers.dense(
        x,
        units=params['n_classes'],
    )
    probabilities = tf.nn.softmax(
        logits,
        name=params['output_name'],
    )
    predicted_classes = tf.argmax(logits, axis=1)

    # 建立預測模型（prediction）
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': probabilities,
        }
        # export_outputs = {
        #     params['output_name']: tf.estimator.export.PredictOutput(probabilities),
        # }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            # export_outputs=export_outputs,
        )

    # 計算損失值及精準度
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
    )

    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=predicted_classes,
        name='accurary_op',
    )
    tf.summary.scalar('accuracy', accuracy[1])  # 紀錄精準度

    # 回傳測試模型（evaluation）
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={ 'accuracy': accuracy },
        )

    # 回傳訓練模型（training）
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()

    # Due to TensorFlow bug https://stackoverflow.com/questions/43234667/tf-layers-batch-normalization-large-test-error
    # batch_normalization() may not work properly. Here is workaround.
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(
            loss,
            global_step=tf.train.get_global_step(),
        )

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op,
    )


def main():
    # 設定紀錄層級，以顯示 TensorFlow 更多資訊
    logging.getLogger().setLevel(logging.INFO)

    # 定義程式參數
    arg_parser = argparse.ArgumentParser(description='使用 TensorFlow 建立模型範例')
    arg_parser.add_argument(
        '--model-base-dir',
        required=True,
        help='模型輸出目錄',
    )
    arg_parser.add_argument(
        '--data-dir',
        required=True,
        help='資料目錄',
    )
    arg_parser.add_argument(
        '--epochs',
        type=int,
        default=64,
        help='訓練回合數',
    )
    arg_parser.add_argument(
        '--output-file',
        default='-',
        help='預測輸出檔案',
    )
    arg_parser.add_argument(
        '--input-width',
        type=int,
        default=48,
        help='模型輸入寬度',
    )
    arg_parser.add_argument(
        '--input-height',
        type=int,
        default=48,
        help='模型輸入高度',
    )
    arg_parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='批次大小',
    )
    arg_parser.add_argument(
        '--input-tensor-name',
        default='input_image',
        help='輸入層名稱',
    )
    arg_parser.add_argument(
        '--output-tensor-name',
        default='probabilities',
        help='輸出層名稱',
    )
    args = arg_parser.parse_args()

    # 常用常數
    input_channel = 3  # 紅綠藍三個通道
    n_classes = 4      # 左傳、右轉、停止、其他四個類別

    # 載入原始資料
    # 習慣上 x 代表輸入資料，y 代表期望標記
    (paths_train, train_x, train_y), (paths_test, test_x) = load_data(
        args.data_dir,
        args.input_height,
        args.input_width,
        input_channel,
        n_classes,
    )
    assert len(set([len(paths_train), train_x.shape[0], train_y.shape[0]])) == 1
    assert len(set([len(paths_test), test_x.shape[0]])) == 1

    # 建立資料輸入函數，訓練集和測試集各一個
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={ X_FEATURE_KEY: train_x },
        y=train_y,
        num_epochs=args.epochs,
        shuffle=True,
        batch_size=args.batch_size,
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={ X_FEATURE_KEY: train_x },
        y=train_y,
        num_epochs=1,
        shuffle=False,
        batch_size=args.batch_size,
    )

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={ X_FEATURE_KEY: test_x },
        y=None,
        num_epochs=1,
        shuffle=False,
        batch_size=args.batch_size,
    )

    input_tensor = tf.placeholder(
        'float',
        shape=[1, args.input_height, args.input_width, input_channel],
        name=args.input_tensor_name,
    )

    serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        { X_FEATURE_KEY: input_tensor },
        default_batch_size=1,
    )

    # 建立模型
    params = {
        'n_classes': n_classes,
        'output_name': args.output_tensor_name,
    }

    custom_classifier = tf.estimator.Estimator(
        model_fn=custom_model_fn,
        params=params,
    )

    # 訓練模型
    custom_classifier.train(
        input_fn=train_input_fn,
    )

    # 測試模型
    eval_results = custom_classifier.evaluate(
        input_fn=eval_input_fn,
    )

    print('損失值\t%f' % eval_results['loss'])
    print('精準度\t%f' % eval_results['accuracy'])

    # 執行預測
    ind2name = [
        'left',   # 0
        'right',  # 1
        'stop',   # 2
        'other',  # 3
    ]

    if test_x.shape[0] > 0:
        predictions = custom_classifier.predict(
            input_fn=test_input_fn,
        )

        if args.output_file == '-':
            for path, pred_dict in zip(paths_test, predictions):
                class_id = pred_dict['class_ids'][0]
                class_name = ind2name[class_id]
                probabilities = pred_dict['probabilities']

                print('路徑\t%s' % path)
                print('預測\t%s' % class_name)
                print('機率\t%s' % probabilities)
                print()
        else:
            with open(args.output_file, 'w') as file_out:
                for path, pred_dict in zip(paths_test, predictions):
                    class_id = pred_dict['class_ids'][0]
                    class_name = ind2name[class_id]
                    probabilities = pred_dict['probabilities']

                    file_out.write('路徑\t%s\n' % path)
                    file_out.write('預測\t%s\n' % class_name)
                    file_out.write('機率\t%s\n' % probabilities)
                    file_out.write('\n')


    # 生成預測用模型檔
    export_dir = custom_classifier.export_savedmodel(
        export_dir_base=args.model_base_dir,
        serving_input_receiver_fn=serving_fn,
    )

    print('模型檔已輸出到', export_dir)


if __name__ == '__main__':
    main()
