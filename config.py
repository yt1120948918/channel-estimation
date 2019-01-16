import warnings


class DefaultConfig:
    data_path = "D:\\data\\channel_estimation\\train\\data_SNR_15"  # 训练数据和验证数据所在文件夹路径
    model_path = "model_file\\model_2_1\\"  # 模型文件保存路径
    summary_path = "log_file\\model_2_1\\"  # summary保存路径，用于tensorboard

    NUM = 12800  # 每个mat文件中的sample数量
    data_num = 1280000  # 训练集数据总量

    batch_size = 128  # 每个batch处理的数据量
    num_epoch = 15  # 总共运行的epoch数量

    gpu_num = "/device:GPU:0"  # 指定用于训练的GPU型号，可填 /device:GPU:0 或者 /device:GPU:1
    per_process_gpu_memory_fraction = 0.8  # 指定占用GPU内存

    optimizer_type = "Adam"  # 优化算法，可以选择用SGD、Momentum或者Adam
    learning_rate = 0.001  # 学习率
    learning_rate_decay = 0.1  # 学习率衰减因子
    learning_rate_change_epoch = 10  # 学习率衰减周期。比如10代表每过10个epoch学习率衰减一次
    momentum = 0.95  # momentum因子

    model_type = 1  # 可填写0、1、2。0表示最原始的反卷积模型，1表示反卷积模型+可学习降维层，2表示反卷积模型+GAN模型
    model_2_layers = 1  # 可学习降维层的层数
    model_2_scale = [0.1, 0.5]

    GAN_type = "DCGAN"  # 可填写"DCGAN"或者"WGAN-GP"，表示模型3中GAN的类型。

    # 可以填写"batch normalization"、"layer normalization"、"weight normalization"和"instance normalization"
    normal_type = "batch normalization"

    def parse(self, kwargs):
        # 根据字典kwargs更新config参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        # 打印配置参数
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


if __name__ == "__main__":
    opt = DefaultConfig()
    model_file = opt.model_path + opt.data_path.split('\\')[-1] + '\\' + opt.data_path.split('\\')[-1] + '.meta'
    print(model_file)
