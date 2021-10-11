# -*- coding:utf-8 -*-
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import umap
import os
from WLLE import WLLE


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return: csr_matrix
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def load_basic_adj_with_threshold(file="raw_dis.pkl", threshold=100, self_w = 1):
    print("loading the plain adj with threshold {}...".format(threshold))
    with open(file, 'rb') as f:
        dis_array, height_arr = pickle.load(file=f)

    # dis_array[dis_array <= threshold] = 1
    dis_array[dis_array > threshold] = 0
    # adj = dis_array - np.eye(dis_array.shape[0])  # remove the self-loop.
    adj = dis_array + self_w * np.eye(dis_array.shape[0])  # add self weight.

    return adj


def get_weight_adj_with_threshold(file="raw_dis.pkl", threshold=100, norm=True):
    # weight = hei / dis
    print("loading the weight adj with threshold {}...".format(threshold))
    with open(file, 'rb') as f:
        dis_arr, height_arr = pickle.load(file=f)

    def calculate_weight(dis, hei):
        if dis > threshold:
            return 0
        elif dis == 0:  # the self-dis.
            return 1
        else:
            return hei / dis

    weight_list = []
    for dis, hei in zip(dis_arr, height_arr):
        weight_list.append(list(map(lambda d, h: calculate_weight(d, h), dis, hei)))

    # same shape as dis_arr
    weight_array = np.array(weight_list) - np.eye(N=dis_arr.shape[0])  # remove the self-loop.

    return RBF_norma(weight_array)


def RBF_norma(arr):
    # convert the 0 to np.nan type, thus we can void the 0 while calculating the mean and var.
    item_list = []
    for row in arr:
        item_list.append(list(map(lambda x: np.nan if x == 0 else x, row)))
    nan_arr = np.array(item_list)
    mean = np.nanmean(nan_arr)
    delta_2 = np.nanvar(nan_arr)

    # calculate the rbf, except the 0 item.
    def cal_rbf(x):
        if x == 0:
            return 0
        else:
            return np.exp(-1 * (abs(x - mean) ** 2 / delta_2))

    rbf_list = []
    for row in arr:
        rbf_list.append(list(map(lambda x: cal_rbf(x), row)))

    return np.array(rbf_list)


def load_mani_adj(file='use_info_left.csv', mode='HLLE', k=50, stable_data=None, show_data_shape=False):
    if stable_data:
        print("loading stable data :{}".format(stable_data))
    else:
        print("loading mainflod adj with mode: {} k:{} ...".format(mode, k))
    mani_data = get_manifold_data(file=file, mode=mode, k=k, stable_data=stable_data)

    # get adj and return.
    adj = pairwise_distances(mani_data)

    if show_data_shape:
        # vis mani_data in picture.
        height = np.loadtxt(fname=file)[:, 2:3].T
        min_hei, max_hei = np.min(height), np.max(height)
        colors = [((i - min_hei) * 1.0) / (max_hei - min_hei) * 256 for i in height]
        plt.scatter(mani_data[:, 0], mani_data[:, 1], marker='o', c=colors[0])
        plt.text(.99, .01, (mode + ': k=%d' % (k)), transform=plt.gca().transAxes,
                 size=10, horizontalalignment='right')
        plt.show()

    return adj


def load_mani_adj_with_threshold(file='use_info_left.csv', mode='HLLE', k=50, dis_file='raw_dis_left.pkl',
                                 threshold=100, stable_data=None):
    if stable_data:
        print("get manifold adj "+stable_data)
    adj_arr = load_mani_adj(file=file, mode=mode, k=k, stable_data=stable_data)
    with open(dis_file, 'rb') as f:
        dis_arr, height_arr = pickle.load(file=f)

    count = 0
    def adj_value_under_threshold(dis, value):
        if dis > threshold:
            return 0
        elif dis == 0:  # the self-dis.
            # return 1
            return threshold
        else:
            return value

    weight_list = []
    for dis, adj in zip(dis_arr, adj_arr):
        weight_list.append(list(map(lambda d, h: adj_value_under_threshold(d, h), dis, adj)))
        # if adj_value_under_threshold(dis, adj) == 0:
        #     count = count + 1

    # weight_adj = np.array(weight_list) - np.eye(N=dis_arr.shape[0])  # remove the self-loop.

    adj = np.array(weight_list)
    # adj = np.divide(1, adj, out=np.zeros(adj.shape),  where=adj!=0)
    # adj = np.divide(200, adj, out=np.zeros(adj.shape),  where=adj!=0)
    adj = np.divide(adj, threshold)

    print("How many edges are cut off under threshold {}".format(threshold))
    print(adj[adj == 0].shape[0] / (adj.shape[0] * adj.shape[0]))
    # adj = np.log(adj)
    # adj = np.exp(-adj+1)
    # adj[adj>1]=1.0
    return adj


def load_features_labels(file="use_info.csv", train_rate=0.5, val_rate=0.3, seq_len=3, pre_len=1, all=False):
    # 获得在一定“窗口”内的沉降量特征 window 指原始数据中用来训练和预测的比例
    data = np.loadtxt(fname=file, delimiter=',', skiprows=1)[:, 3:]
    data = data.transpose()  # shape=(time_sequence, number_of_nodes)
    print("Shape of {}: {} ".format(file, data.shape))
    num_nodes = data.shape[1]
    max_value = np.max(data)
    data_normal = data / max_value

    train_size = int(data_normal.shape[0] * train_rate)
    val_size = int(data_normal.shape[0] * val_rate)
    train_data = data_normal[0: train_size]
    val_data = data_normal[train_size: train_size + val_size]
    test_data = data_normal[train_size + val_size:]

    def get_X_and_Y(data):
        X, Y = [], []
        for i in range(len(data) - seq_len - pre_len + 1):
            X.append(data[i: i + seq_len])
            Y.append(data[i + seq_len: i + seq_len + pre_len])
        return np.array(X), np.array(Y)

    trainX, trainY = get_X_and_Y(train_data)
    valX, valY = get_X_and_Y(val_data)
    testX, testY = get_X_and_Y(test_data)
    if all:
        all_X, all_Y = get_X_and_Y(data_normal)

    print("Shape of trainX: {} trainY: {}".format(trainX.shape, trainY.shape))
    print("Shape of valX: {} valY: {}".format(valX.shape, valY.shape))
    print("Shape of testX: {} testY: {}".format(testX.shape, testY.shape))

    if all:
        return num_nodes, max_value, trainX, trainY, valX, valY, testX, testY, all_X, all_Y
    else:
        return num_nodes, max_value, trainX, trainY, valX, valY, testX, testY


def get_manifold_data(file="data/use_info_left.csv", mode='LLE', k=None, stable_data=None):
    # stable_data = None
    if stable_data:
        trans_data = np.loadtxt(stable_data, delimiter=',')
        return trans_data

    # get info from lon lat and height, and turn into 2 components
    train_data = np.loadtxt(fname=file)[:, 0:3]
    if mode == "LLE":
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                     method='standard').fit_transform(train_data)
    elif mode == "HLLE":
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                     method='hessian').fit_transform(train_data)
    elif mode == "MLLE":
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                     method='modified').fit_transform(train_data)
    elif mode == "LTSA":
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                     method='ltsa').fit_transform(train_data)
    elif mode == "T-SNE":
        trans_data = manifold.TSNE(n_components=2, init='pca', random_state=77).fit_transform(train_data)
    elif mode == 'Isomap':
        trans_data = manifold.Isomap(n_neighbors=k, n_components=2).fit_transform(train_data)
    elif mode == 'MDS':
        trans_data = manifold.MDS(n_components=2).fit_transform(train_data)
    elif mode == 'SE':
        trans_data = manifold.SpectralEmbedding(n_components=2).fit_transform(train_data)
    elif mode == 'umap':
        reducer = umap.UMAP(n_neighbors=k, n_components=2, metric='euclidean', random_state=42)
        reducer.fit(train_data)
        trans_data = reducer.transform(train_data)
    return trans_data


def data_vis(file="data/use_info_left.csv"):
    data = np.loadtxt(fname=file)[:, 3:]
    history, label = data[:, 0: -1], data[:, -1:]

    for x, y in zip(history, label):
        # plot = show_plot([x[0], y[0], simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
        plot = show_plot([x, y, np.mean(x)], 'Simple Mean model')
        plot.show()


def show_plot(plot_data, title):
    import matplotlib.pyplot as plt
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = range(1, len(plot_data[0]) + 1)
    time_pre = len(plot_data[0]) + 1

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(time_pre, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xticks([1, 5, 10, 15, 20])
    plt.xlabel('Time-Step')
    return plt


if __name__ == '__main__':
    # get and save manifold image, csv
    a = ['LLE', 'MLLE', 'Isomap', 'T-SNE', 'UMAP']
    b = ['HLLE', 'LTSA']
    c = ['no']
    d = ['WLLE']

