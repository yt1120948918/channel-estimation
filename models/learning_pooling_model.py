import tensorflow as tf
from utils import model_utils
from models.base_model import BaseModel


class LearningPoolingModel(BaseModel):
    def __init__(self, batch_size, normal_type, is_training, model_2_layers, name):
        self.batch_size = batch_size
        self.normal_type = normal_type
        self.is_training = is_training
        self.model_2_layers = model_2_layers
        self.name = name
        super(LearningPoolingModel, self).__init__()

    def calculate_loss(self, inputs, labels, scale):
        pred_1 = self.create_model(inputs)
        pred_2 = self.create_model(labels)

        if len(pred_1) != len(scale):
            raise Exception("请重新设置超参数model_2_scale")

        train_loss = 0
        for i in range(len(pred_1)):
            train_loss += scale[i] * tf.reduce_sum((pred_1[i] - pred_2[i]) ** 2) / self.batch_size

        return train_loss

    def create_model(self, inputs):
        with tf.variable_scope(self.name):
            if self.model_2_layers == 1:
                # 第一层
                conv1_1 = model_utils.conv_layer(inputs, 2, 8, [3, 3], [1, 1, 1, 1], 'SAME', None,
                                                 self.is_training, 'conv1_1')
                conv1_1 = model_utils.leaky_relu(conv1_1)
                conv1_1 = model_utils.avg_pool(conv1_1, [1, 2, 2, 1], [1, 2, 1, 1], 'VALID', name='conv1_1')

                return [inputs, conv1_1]

            elif self.model_2_layers == 2:
                # 第一层
                conv1_1 = model_utils.conv_layer(inputs, 2, 8, [3, 3], [1, 1, 1, 1], 'SAME', None,
                                                 self.is_training, 'conv1_1')
                conv1_1 = model_utils.leaky_relu(conv1_1)
                conv1_1 = model_utils.avg_pool(conv1_1, [1, 2, 2, 1], [1, 2, 1, 1], 'VALID', 'conv1_1')

                # 第二层
                conv2_1 = model_utils.conv_layer(conv1_1, 8, 16, [3, 3], [1, 1, 1, 1], 'SAME', None,
                                                 self.is_training, 'conv2_1')
                conv2_1 = model_utils.leaky_relu(conv2_1)
                conv2_1 = model_utils.avg_pool(conv2_1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', 'conv2_1')

                return [inputs, conv1_1, conv2_1]

            elif self.model_2_layers == 3:
                # 第一层
                conv1_1 = model_utils.conv_layer(inputs, 2, 8, [3, 3], [1, 1, 1, 1], 'SAME', None,
                                                 self.is_training, 'conv1_1')
                conv1_1 = model_utils.leaky_relu(conv1_1)
                conv1_1 = model_utils.avg_pool(conv1_1, [1, 2, 2, 1], [1, 2, 1, 1], 'VALID', name='conv1_1')

                # 第二层
                conv2_1 = model_utils.conv_layer(conv1_1, 8, 16, [3, 3], [1, 1, 1, 1], 'SAME', None,
                                                 self.is_training, 'conv2_1')
                conv2_1 = model_utils.leaky_relu(conv2_1)
                conv2_1 = model_utils.avg_pool(conv2_1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='conv2_1')

                # 第三层
                conv3_1 = model_utils.conv_layer(conv2_1, 16, 32, [3, 3], [1, 1, 1, 1], 'SAME', None,
                                                 self.is_training, 'conv3_1')
                conv3_1 = model_utils.leaky_relu(conv3_1)
                conv3_1 = model_utils.avg_pool(conv3_1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='conv3_1')

                return [inputs, conv1_1, conv2_1, conv3_1]
