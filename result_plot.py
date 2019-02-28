import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import spline


x = np.array([10, 12, 15, 17, 22, 30, 40])
traditional = [200.3, 133.7, 76.7, 55.9, 31.3, 21.7, 20.2]
base = [207.2, 140.2, 83.1, 56.7, 27.9, 11.5, 5.2]
learning_pool_1 = [203.2, 135.1, 75.0, 54.3, 26.8, 10.1, 3.9]
learning_pool_2 = [201.0, 133.0, 74.0, 53.1, 25.9, 9.2, 3.0]
learning_pool_3 = [202.1, 134.1, 74.5, 54.0, 27.0, 9.9, 3.7]

plt.figure()
legends = []
# plt.plot(x, traditional, color='red', linestyle='solid')
plt.plot(x, base, color='blue', linestyle='dashed')
plt.plot(x, learning_pool_1, color='green', linestyle='dashdot')
plt.plot(x, learning_pool_2, color='yellow', linestyle='dashdot')
plt.plot(x, learning_pool_3, color='black', linestyle='dashdot')
# legends.append('traditional_model')
legends.append('base_model')
legends.append('learning_pool_model_1')
legends.append('learning_pool_model_1')
legends.append('learning_pool_model_1')
plt.xlabel('SNR')
plt.ylabel('MSE')
plt.legend(legends)
plt.savefig("result.png")
plt.show()
plt.close()
