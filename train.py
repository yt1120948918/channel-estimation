# import os
import time
import tensorflow as tf
from config import DefaultConfig
from models import GModel
from models import DModel
from models import LearningPoolingModel
from data.dataset import CSISet
from utils import utils
from utils import model_utils


def train(**kwargs):
    # 根据命令行参数更新配置
    opt = DefaultConfig()
    opt.parse(kwargs)
    print("参数配置完成")

    # 优化器
    learning_rate = opt.learning_rate
    # optimizer默认是Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.5,
                                       beta2=0.9)
    if opt.optimizer_type == "SGD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif opt.optimizer_type == "Momentum":
        momentum = opt.momentum
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=momentum)
    elif opt.optimizer_type == "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.5,
                                           beta2=0.9)

    # 建立静态图
    with tf.Graph().as_default():
        with tf.name_scope("inputs"):
            inputs = tf.placeholder("float", [None, 24, 2, 2], name="model_input")
            labels = tf.placeholder("float", [None, 72, 14, 2], name="labels")

        # 定义模型，统计并分类需要训练的模型参数
        model = []
        if opt.model_type == 1:  # 反卷积
            gmodel = GModel(opt.batch_size, opt.normal_type, True, "generate_model")
            model.append(gmodel)
        elif opt.model_type == 2:  # 反卷积+可学习pooling
            gmodel = GModel(opt.batch_size, opt.normal_type, True, "generate_model")
            model.append(gmodel)
            learningpoolingmodel = LearningPoolingModel(opt.batch_size, opt.normal_type, True, opt.model_2_layers,
                                                        "learning_pooling_model")
            model.append(learningpoolingmodel)
        elif opt.model_type == 3:  # 反卷积+GAN
            gmodel = GModel(opt.batch_size, opt.normal_type, True, "generate_model")
            model.append(gmodel)
            dmodel = DModel(opt.batch_size, opt.normal_type, True, opt.GAN_type, "discriminate_model")
            model.append(dmodel)
        # print(model)

        # 统计并分类需要训练的参数
        # 由于下面加上了对tf.GraphKeys.UPDATE_OPS的依赖，所以get_vars函数要加到calculate_loss函数后面
        # 不然就会导致all_vars为空
        def get_vars():
            all_vars = tf.trainable_variables()
            # print(all_vars)
            gg_vars = [var for var in all_vars if "generate_model" in var.name]
            dd_vars = [var for var in all_vars if "discriminate_mode" in var.name]
            ll_pp_vars = [var for var in all_vars if "learning_pooling_model" in var.name]
            return gg_vars, dd_vars, ll_pp_vars

        # 加上对update_ops的依赖，不然BN就会出现问题！
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.device(opt.gpu_num):
            if opt.model_type == 1:  # 反卷积
                pre_loss, mse, pred = model[0].calculate_loss(inputs, labels)
                g_vars, _, _ = get_vars()
                with tf.control_dependencies(update_ops):
                    train_ops = optimizer.minimize(pre_loss, var_list=g_vars)
            elif opt.model_type == 2:  # 反卷积+可学习pooling
                _, mse, pred = model[0].calculate_loss(inputs, labels)
                l_p_loss = model[1].calculate_loss(pred, labels, opt.model_2_scale)
                g_vars, _, l_p_vars = get_vars()
                with tf.control_dependencies(update_ops):
                    train_ops = optimizer.minimize(l_p_loss, var_list=g_vars + l_p_vars)
            elif opt.model_type == 3:  # 反卷积+GAN
                pre_loss, mse, pred = model[0].calculate_loss(inputs, labels)
                gen_loss, dis_loss = model[1].calculate_loss(pred, labels)
                g_vars, d_vars, _ = get_vars()
                with tf.control_dependencies(update_ops):
                    # D网络的训练 --> G网络的训练 ——> 先验网络（也就是G网络）的训练
                    d_train_ops = optimizer.minimize(dis_loss, var_list=d_vars)
                    g_train_ops = optimizer.minimize(gen_loss, var_list=g_vars)
                    pre_train_ops = optimizer.minimize(pre_loss, var_list=g_vars)

        tf.summary.scalar("MSE", mse)

        tf.add_to_collection("input_batch", inputs)
        tf.add_to_collection("predictions", pred)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # 开始训练
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = opt.per_process_gpu_memory_fraction
        with tf.Session(config=config) as sess:
            # 首先是参数的初始化
            sess.run(init)

            if opt.model_type == 1:
                model_type = "model_1"
            elif opt.model_type == 2:
                model_type = "model_2_" + str(opt.model_2_layers)
            elif opt.model_type == 3:
                model_type = "model_3"
            summary_path = opt.summary_path + model_type + "\\data_SNR_" + str(opt.SNR)
            writer = tf.summary.FileWriter(summary_path, sess.graph)
            merge_ops = tf.summary.merge_all()

            start = time.time()

            data_path = opt.train_data_path + "data_SNR_" + str(opt.SNR)
            # 定义训练集dataset
            train_dataset = CSISet(data_path, opt.batch_size, True, state="train")
            # 定义验证集dataset
            validation_dataset = CSISet(data_path, opt.batch_size, True, state="validation")

            # 保存训练集和验证集的中间值，用于后续的画图
            train_mse_for_plot = []
            valid_mse_for_plot = []

            for num in range(opt.num_epoch):
                # 判断是否需要改变学习率
                if opt.optimizer_type == "Momentum" and (num % opt.learning_rate_change_epoch) == 0:
                    learning_rate *= opt.learning_rate_decay
                    print("第%i个epoch开始，当前学习率是%f" % (num, learning_rate))

                for ii, (batch_x, batch_y) in enumerate(train_dataset.get_data()):
                    if opt.model_type == 1 or opt.model_type == 2:
                        _, train_mse, summary = sess.run([train_ops, mse, merge_ops],
                                                         feed_dict={inputs: batch_x,
                                                                    labels: batch_y})
                    elif opt.model_type == 3:
                        _, _, _, train_mse, summary = sess.run([d_train_ops, g_train_ops, pre_train_ops,
                                                                mse, merge_ops],
                                                               feed_dict={inputs: batch_x,
                                                                          labels: batch_y})
                    writer.add_summary(summary)

                    if (ii + 1) % 1000 == 0:
                        print("epoch-%d, batch_num-%d: 当前batch训练数据误差是%f" % (num + 1, ii + 1, train_mse))

                        # 每1000个batch就在验证集上测试一次
                        validate_mse = 0
                        jj = 1
                        for (validate_x, validate_y) in validation_dataset.get_data():
                            temp_mse = sess.run(mse, feed_dict={inputs: validate_x,
                                                                labels: validate_y})
                            validate_mse += temp_mse
                            jj += 1
                        validate_mse = validate_mse / (jj + 1)
                        print("epoch-%d: 当前阶段验证集数据平均误差是%f" % (num + 1, validate_mse))
                        train_mse_for_plot.append(train_mse)
                        valid_mse_for_plot.append(validate_mse)

            end = time.time()

            utils.print_time(start, end, "跑完" + str(opt.num_epoch) + "个epoch")

            plot_path = opt.result_path + model_type + "\\data_SNR_" + str(opt.SNR) + "\\train"
            utils.plot_fig(train_mse_for_plot, valid_mse_for_plot, plot_path)
            print("训练过程中最小验证误差是%f" % min(valid_mse_for_plot))

            # 保存模型文件
            model_file = opt.model_path + model_type + "\\data_SNR_" + str(opt.SNR) + "\\data_SNR_" + str(opt.SNR)
            model_utils.save_model(saver, sess, model_file)


if __name__ == "__main__":
    train()
