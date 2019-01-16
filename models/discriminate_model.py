import tensorflow as tf
from utils import model_utils
from models.base_model import BaseModel


LAMBDA = 10


class DModel(BaseModel):
    def __init__(self, batch_size, normal_type, is_training, gan_type, name):
        self.batch_size = batch_size
        self.normal_type = normal_type
        self.is_training = is_training
        self.GAN_type = gan_type
        self.name = name
        super(DModel, self).__init__()

    def calculate_loss(self, inputs, labels):
        fake_pred = self.create_model(inputs)
        real_pred = self.create_model(labels)

        if self.GAN_type == "DCGAN":
            # G网络对应损失函数
            gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_pred,
                                                                              labels=tf.ones_like(fake_pred)))
            # D网络对应损失函数
            dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_pred,
                                                                                   labels=tf.zeros_like(fake_pred)))
            dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_pred,
                                                                                   labels=tf.ones_like(real_pred)))
            dis_loss = dis_loss_fake + dis_loss_real
            return gen_loss, dis_loss
        elif self.GAN_type == "WGAN-GP":
            # WGAN模型对应的损失函数
            gen_loss = -tf.reduce_mean(fake_pred)
            dis_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)

            # 梯度惩罚（Gradient penalty）
            alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
            differences = inputs - labels
            interpolates = inputs + alpha * differences
            gradients = tf.gradients(self.create_model(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            dis_loss += LAMBDA * gradient_penalty
            return gen_loss, dis_loss

    def create_model(self, inputs):
        with tf.variable_scope(self.name):
            # 模型大小变化如下：
            # 72x14x2      -(3x3,8,(2,1),'SAME')->
            # 36x14x8      -(3x3,16,(1,1),'VALID')->
            # 34x12x16     -(3x3,32,(1,1),'VALID')->
            # 32x10x32     -(3x3,64,(1,1),'VALID')->
            # 30x8x64      -(3x3,128,(1,1),'VALID')->
            # 28x6x128     -(3x3,256,(1,1),'VALID')->
            # 26x4x256     -(3x3,2,(1,1),'VALID')->
            # 24x2x2       -reshape([-1])->
            # [96]  最后输出的是一个长度为96的向量
            # 第一层
            conv1_1 = model_utils.conv_layer(inputs, 2, 8,
                                             [3, 3], [1, 2, 1, 1], 'SAME', self.normal_type,
                                             self.is_training, 'conv1_1')
            conv1_1 = model_utils.leaky_relu(conv1_1)
            # 第二层
            conv2_1 = model_utils.conv_layer(conv1_1, 8, 16,
                                             [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                             self.is_training, 'conv2_1')
            conv2_1 = model_utils.leaky_relu(conv2_1)
            # 第三层
            conv3_1 = model_utils.conv_layer(conv2_1, 16, 32,
                                             [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                             self.is_training, 'conv3_1')
            conv3_1 = model_utils.leaky_relu(conv3_1)
            # 第四层
            conv4_1 = model_utils.conv_layer(conv3_1, 32, 64,
                                             [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                             self.is_training, 'conv4_1')
            conv4_1 = model_utils.leaky_relu(conv4_1)
            # 第五层
            conv5_1 = model_utils.conv_layer(conv4_1, 64, 128,
                                             [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                             self.is_training, 'conv5_1')
            conv5_1 = model_utils.leaky_relu(conv5_1)
            # 第六层
            conv6_1 = model_utils.conv_layer(conv5_1, 128, 256,
                                             [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                             self.is_training, 'conv6_1')
            conv6_1 = model_utils.leaky_relu(conv6_1)
            # 第七层
            conv7_1 = model_utils.conv_layer(conv6_1, 256, 2,
                                             [3, 3], [1, 1, 1, 1], 'VALID', self.normal_type,
                                             self.is_training, 'conv7_1')
            # conv7_1 = model_utils.leaky_relu(conv7_1)
            outputs = tf.reshape(conv7_1, [-1])
            outputs = model_utils.linear(outputs, 1, 'linear8_1')

            return outputs
