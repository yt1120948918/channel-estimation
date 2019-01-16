import numpy as np
import tensorflow as tf
from utils import model_utils
from models.base_model import BaseModel


class GModel(BaseModel):
    def __init__(self, batch_size, normal_type, is_training, name):
        self.batch_size = batch_size
        self.normal_type = normal_type
        self.is_training = is_training
        self.name = name
        super(GModel, self).__init__()

    def calculate_loss(self, inputs, labels, is_pred_return=False):
        pred = self.create_model(inputs)
        train_loss = tf.reduce_sum((pred - labels) ** 2) / self.batch_size

        # 将pred的某些位置的值改为0
        shape = pred.get_shape()
        mask = np.ones((shape[0].value, shape[1].value, shape[2].value, shape[3].value))
        mask[:, 0:5, 6:8, :] = 0
        mask[:, 67:72, 6:8, :] = 0
        new_pred = pred * mask
        mse = tf.reduce_sum((new_pred - labels) ** 2) / self.batch_size

        if is_pred_return:
            return train_loss, mse, pred
        return train_loss, mse

    def create_model(self, inputs):
        with tf.name_scope(self.name):
            # 模型大小变化如下：
            # 24x2x2       -(3x3,256,(1,1),'VALID')->
            # 26x4x256     -(3x3,128,(1,1),'VALID')->
            # 28x6x128     -(3x3,64,(1,1),'VALID')->
            # 30x8x64     -(3x3,32,(1,1),'VALID')->
            # 32x10x32     -(3x3,16,(1,1),'VALID')->
            # 34x12x16     -(3x3,8,(1,1),'VALID')->
            # 36x14x8     -(3x3,4,(2,1),'SAME')->
            # 72x14x4     -(1x1,2,(1,1),'SAME')->
            # 72x14x2
            # 参数output_size--[batch_size, height, width, channels]
            # 第一层
            deconv1_1 = model_utils.deconv_layer(inputs, [self.batch_size, 26, 4, 256], 2, 256,
                                                 [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                                 self.is_training, 'deconv1_1')
            deconv1_1 = tf.nn.relu(deconv1_1)
            # 第二层
            deconv2_1 = model_utils.deconv_layer(deconv1_1, [self.batch_size, 28, 6, 128], 256, 128,
                                                 [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                                 self.is_training, 'deconv2_1')
            deconv2_1 = tf.nn.relu(deconv2_1)
            # 第三层
            deconv3_1 = model_utils.deconv_layer(deconv2_1, [self.batch_size, 30, 8, 64], 128, 64,
                                                 [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                                 self.is_training, 'deconv3_1')
            deconv3_1 = tf.nn.relu(deconv3_1)
            # 第四层
            deconv4_1 = model_utils.deconv_layer(deconv3_1, [self.batch_size, 32, 10, 32], 64, 32,
                                                 [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                                 self.is_training, 'deconv4_1')
            deconv4_1 = tf.nn.relu(deconv4_1)
            # 第五层
            deconv5_1 = model_utils.deconv_layer(deconv4_1, [self.batch_size, 34, 12, 16], 32, 16,
                                                 [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                                 self.is_training, 'deconv5_1')
            deconv5_1 = tf.nn.relu(deconv5_1)
            # 第六层
            deconv6_1 = model_utils.deconv_layer(deconv5_1, [self.batch_size, 36, 14, 8], 16, 8,
                                                 [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                                 self.is_training, 'deconv6_1')
            deconv6_1 = tf.nn.relu(deconv6_1)
            # 第七层
            deconv7_1 = model_utils.deconv_layer(deconv6_1, [self.batch_size, 72, 14, 4], 8, 4,
                                                 [3, 3], [1, 2, 1, 1], 'SAME', self.normal_type,
                                                 self.is_training, 'deconv7_1')
            deconv7_1 = tf.nn.relu(deconv7_1)
            # 最后的输出层normal_type设置为None，仅仅是利用1x1的卷积核降维
            outputs = model_utils.deconv_layer(deconv7_1, [self.batch_size, 72, 14, 2], 4, 2,
                                               [1, 1], [1, 1, 1, 1], 'SAME', None, self.is_training, 'outputs')

            return outputs
