import time
import tensorflow as tf
from config import DefaultConfig
from data.dataset import CSISet


def inference(**kwargs):
    # 根据命令行参数更新配置
    opt = DefaultCongfig()
    opt.parse(kwargs)
    print("参数配置完成")

    # 加载静态图
    model_files = opt.model_path + opt.data_path.split('\\')[-1]
    saver = tf.train.import_meta_graph(model_files + '\\' + opt.data_path.split('\\')[-1] + '.meta')

    # 开始测试
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = opt.per_process_gpu_memory_fraction
    with tf.Session(config=config) as sess:
    	# 加载参数值
    	saver.restore(sess, tf.train.latest_checkpoint(model_files))

    	start_time = time.time()

    	# 定义测试集dataset
    	test_dataset = CSISet(opt.data_path, opt.batch_size, False, "test")
