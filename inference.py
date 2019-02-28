import ipdb
import time
import tensorflow as tf
import numpy as np
from config import DefaultConfig
from data.dataset import CSISet
from utils.utils import print_time


def inference(**kwargs):
    # 根据命令行参数更新配置
    opt = DefaultConfig()
    opt.parse(kwargs)
    print("参数配置完成")

    if opt.model_type == 1:
        model_type = "model_1"
    elif opt.model_type == 2:
        model_type = "model_2_" + opt.model_2_layers
    elif opt.model_type == 3:
        model_type = "model_3"

    # 加载静态图
    model_files = opt.model_path + model_type + "\\data_SNR_" + str(opt.SNR)
    saver = tf.train.import_meta_graph(model_files + "\\data_SNR_" + str(opt.SNR) + ".meta")

    # 开始测试
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = opt.per_process_gpu_memory_fraction
    with tf.Session(config=config) as sess:
        # 加载参数值
        saver.restore(sess, tf.train.latest_checkpoint(model_files))

        # 定义测试集dataset
        test_dataset = CSISet(opt.test_data_path, opt.batch_size, False, "test")

        data_loss = []  # 保存每个batch的发送信号和预测的发送信号之间的误差

        print("开始预测过程！")
        start_time = time.time()

        for ii, (batch_x, batch_tx, batch_rx) in enumerate(test_dataset.get_data()):
            inputs = tf.get_collection("input_batch")[0]
            predictions = tf.get_collection("predictions")[0]

            # pred_H是模型预测的信道完整特性，维度是[batch, 72, 14, 2]
            # 再利用公式 H=rx/tx 和(pred_H, batch_rx)就可以得到pred_rx
            pred_H = np.squeeze(np.array(sess.run([predictions],
                                                  feed_dict={inputs: batch_x})), axis=0)
            complex_pred_H = pred_H[:, :, :, 0] + pred_H[:, :, :, 1] * 1j
            # ipdb.set_trace()
            pred_batch_tx = np.divide(batch_rx, complex_pred_H)

            pred_batch_tx[:, :5, 5:7] = 1.
            pred_batch_tx[:, 67:72, 5:7] = 1.
            batch_tx[:, :5, 5:7] = 1.
            batch_tx[:, 67:72, 5:7] = 1.

            batch_data_loss_ratio = np.mean(np.divide(abs(pred_batch_tx - batch_tx), abs(batch_tx)))
            # print(batch_data_loss)
            print("第%d个batch的发送信息预测平均误差是%.6f" % (ii+1, batch_data_loss_ratio))
            data_loss.append(batch_data_loss_ratio)

        result = np.mean(data_loss)
        print("信噪比为%d时模型在测试集上的平均估计误差为%.2f" % (opt.SNR, result))

        end_time = time.time()
        print_time(start_time, end_time, "整个测试过程")

        result_path = opt.result_path + model_type + "\\data_SNR_" + str(opt.SNR) + "\\test\\result.npy"
        np.save(result_path, data_loss)


if __name__ == "__main__":
    inference()
