import warnings


class DefaultConfig:
    SNR = 10  # 可填10、12、15、17、22、30、40、60、80、100。
    train_data_path = "D:\\data\\channel_estimation\\train\\"  # 训练数据和验证数据所在文件夹路径
    test_data_path = "D:\\data\channel_estimation\\test\\pilot_48\\data_for_NN\\LS\\"  # 测试数据所在文件夹路径
    model_path = "model_file\\"  # 模型文件保存路径
    summary_path = "log_file\\"  # summary保存路径，用于tensorboard
    result_path = "result\\"  # 结果保存路径
    model_type = 3  # 可填写1、2、3。1表示最原始的反卷积模型，2表示反卷积模型+可学习降维层，3表示反卷积模型+GAN模型

    NUM = 12800  # 每个mat文件中的sample数量
    data_num = 1280000  # 训练集数据总量

    batch_size = 128  # 每个batch处理的数据量
    num_epoch = 35  # 总共运行的epoch数量

    gpu_num = "/device:GPU:0"  # 指定用于训练的GPU型号，可填 /device:GPU:0 或者 /device:GPU:1
    per_process_gpu_memory_fraction = 0.8  # 指定占用GPU内存

    # 注：如果model_type选择了3，那么这里优化算法最好选择SGD，学习率也尽可能的小
    # 不然很容易因为梯度过大而导致模型输出的损失为NaN
    optimizer_type = "SGD"  # 优化算法，可以选择用SGD、Momentum或者Adam
    learning_rate = 0.00000001  # 学习率

    learning_rate_decay = 0.1  # 学习率衰减因子
    learning_rate_change_epoch = 10  # 学习率衰减周期。比如10代表每过10个epoch学习率衰减一次
    momentum = 0.95  # momentum因子

    model_2_layers = 3  # 可学习降维层的层数，可填1、2、3。
    model_2_scale = [0.1, 0.5, 0.8, 1.]

    GAN_type = "WGAN-GP"  # 可填写"DCGAN"或者"WGAN-GP"，表示模型3中GAN的类型。

    # 可以填写"batch normalization"、"layer normalization"、"weight normalization"、"instance normalization"和"None"
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
