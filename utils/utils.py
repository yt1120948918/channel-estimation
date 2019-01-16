import numpy as np
import matplotlib.pyplot as plt


def print_time(start_time, end_time, des="整个过程"):
    spend_time = end_time - start_time
    if spend_time < 60:
        print(des + "耗时 %.2f 秒" % spend_time)
    elif 60 <= spend_time < 3600:
        print(des + "耗时 %.2f 分钟" % (spend_time / 60))
    elif spend_time >= 3600:
        print(des + "耗时 %.2f 小时" % (spend_time / 3600))


def plot_fig(train_results, valid_results, path):
    save_path = path + "\\train-valid-mse.jpg"
    x = np.arange(1, len(train_results) + 1)
    plt.figure()
    legends = []
    plt.plot(x, train_results, color="red", linestyle="solid")
    plt.plot(x, valid_results, color="blue", linestyle="dashed")
    legends.append("train_mse")
    legends.append("valid_mse")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(legends)
    plt.savefig(save_path)
    plt.close()
