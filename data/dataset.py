import os
import numpy as np
import scipy.io as scio


class CSISet:
    def __init__(self, path, batch_size, is_shuffle=True, state="train"):
        self.path = path
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.state = state

    def get_data(self):
        # 这里的代码最好结合"\data\数据目录结构.txt"查看
        if self.state == "train":
            x_files = [file for file in os.listdir(self.path + "\\x") if "validation" not in file]
            y_files = [file for file in os.listdir(self.path + "\\y") if "validation" not in file]
            indices_1 = np.array(range(len(x_files)))
            if self.is_shuffle:
                indices_1 = np.random.permutation(len(x_files))

            for i in indices_1:
                x_file_path = self.path + "\\x\\" + x_files[i]
                y_file_path = self.path + "\\y\\" + y_files[i]
                x_data = scio.loadmat(x_file_path)
                y_data = scio.loadmat(y_file_path)
                train_data_x = np.float32(np.transpose(x_data['h_data'], axes=[3, 0, 1, 2]))  # h
                train_data_y = np.float32(np.transpose(y_data['H_data'], axes=[3, 0, 1, 2]))  # H
                if self.is_shuffle:
                    indices_2 = np.random.permutation(train_data_x.shape[0])
                    train_data_x = train_data_x[indices_2, :, :, :]
                    train_data_y = train_data_y[indices_2, :, :, :]
                num_steps = int(train_data_x.shape[0] / self.batch_size)
                for j in range(num_steps):
                    batch_x = train_data_x[self.batch_size*j:self.batch_size*(j+1), :, :, :]
                    batch_y = train_data_y[self.batch_size*j:self.batch_size*(j+1), :, :, :]

                    yield batch_x, batch_y
        elif self.state == "validation":
            x_files = [file for file in os.listdir(self.path + "\\x") if "validation" in file]
            y_files = [file for file in os.listdir(self.path + "\\y") if "validation" in file]
            indices_1 = np.array(range(len(x_files)))
            if self.is_shuffle:
                indices_1 = np.random.permutation(len(x_files))

            for i in indices_1:
                x_file_path = self.path + "\\x\\" + x_files[i]
                y_file_path = self.path + "\\y\\" + y_files[i]
                x_data = scio.loadmat(x_file_path)
                y_data = scio.loadmat(y_file_path)
                train_data_x = np.float32(np.transpose(x_data['h_data'], axes=[3, 0, 1, 2]))  # h
                train_data_y = np.float32(np.transpose(y_data['H_data'], axes=[3, 0, 1, 2]))  # H
                if self.is_shuffle:
                    indices_2 = np.random.permutation(train_data_x.shape[0])
                    train_data_x = train_data_x[indices_2, :, :, :]
                    train_data_y = train_data_y[indices_2, :, :, :]
                num_steps = int(train_data_x.shape[0] / self.batch_size)
                for j in range(num_steps):
                    batch_x = train_data_x[self.batch_size * j:self.batch_size * (j + 1), :, :, :]
                    batch_y = train_data_y[self.batch_size * j:self.batch_size * (j + 1), :, :, :]

                    yield batch_x, batch_y
        elif self.state == "test":
            files = os.listdir(self.path)
            for file in files:
                x_file_path = self.path + "\\" + file + "\\x\\h_data_for_channel_estimation_test.mat"
                y_file_path = self.path + "\\" + file + "\\y\\H_data_for_channel_estimation_test.mat"
                x_data = scio.loadmat(x_file_path)
                y_data = scio.loadmat(y_file_path)
                train_data_x = np.float32(np.transpose(x_data['h_data'], axes=[3, 0, 1, 2]))  # h
                train_data_y = np.float32(np.transpose(y_data['H_data'], axes=[3, 0, 1, 2]))  # H
                if self.is_shuffle:
                    indices_2 = np.random.permutation(train_data_x.shape[0])
                    train_data_x = train_data_x[indices_2, :, :, :]
                    train_data_y = train_data_y[indices_2, :, :, :]
                num_steps = int(train_data_x.shape[0] / self.batch_size)
                for j in range(num_steps):
                    batch_x = train_data_x[self.batch_size * j:self.batch_size * (j + 1), :, :, :]
                    batch_y = train_data_y[self.batch_size * j:self.batch_size * (j + 1), :, :, :]

                    yield batch_x, batch_y
