import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from sklearn import manifold
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import umap
import os
import tensorflow as tf
from utils import evaluation, parse_argus
import time
from TERME import TERME
# import scipy
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from load_data import load_features_labels, get_weight_adj_with_threshold, \
    load_mani_adj, data_vis, load_mani_adj_with_threshold, load_basic_adj_with_threshold
from matplotlib import cm
from matplotlib.font_manager import FontProperties, rcParams


def get_color(color='rgb', data_set='1'):
    """

    Parameters
    ----------
    color = rgb or height

    Returns color matrix of shape=(num_samples, RGB_channels)
    -------

    """
    color_file = "data/YC0{}_rel.csv".format(data_set)
    data = np.loadtxt(fname=color_file, delimiter=',', skiprows=1)[:, 0:3]
    if color == 'rgb':
        # using original color as the rgb tuple
        colors = np.zeros((3, data.shape[0]))
        for i in range(3):
            h = data[:, i]
            temp = (h - h.min()) / (h.max() - h.min())
            # temp = (h) / (h.max() - h.min())
            # temp = (h - np.mean(h, axis=0)) / np.std(h, axis=0)
            colors[i] = temp
        colors = colors.T
    elif color == 'height':
        height = data[:, 2]
        min_hei, max_hei = np.min(height), np.max(height)
        colors = [((i - min_hei) * 1.0) / (max_hei - min_hei) * 256 for i in height]
        # colors = colors.T
        # colors = np.tile(colors, (3)).reshape([len(colors), -1])
    elif color == 'cm':
        colors = cm.get_cmap(name='Blues')(data[:, 2])
        # print(colors)
    return colors


def draw_box_plot(file=r"D:\projects\T-GCN-master\saved_model\pure_result.csv"):
    #   ts_rmse	ts_mae	ts_acc	ts_r2	ts_var	name	time_stamp	type
    data = pd.read_csv(file)
    embedding_metrics = ['T-SNE', 'Isomap', 'UMAP', 'LTSA', 'HLLE', 'MLLE', 'LLE', 'TERME_300']
    predict_methods = ['TGCN', 'HA', 'ARIMA', 'SVR', 'GRU ', 'LSTM']
    tgcn_parameter = ['TGCN-10units', 'TGCN-30units', 'TGCN-40units', 'TGCN-50units']
    rnn_parameter = ['gru300', 'gru2-300', 'lstm300', 'lstm2-300']
    umap = ['UMAP10', 'UMAP15', 'UMAP20', 'UMAP30', 'UMAP50']
    TERME = ['TERME_100', 'TERME_200', 'TERME_300', 'TERME_400', 'TERME_450', 'TERME_850']

    print(embedding_metrics)
    type_list = embedding_metrics
    box_list = []
    box_list2 = []

    for i in range(len(type_list)):
        box_list.append(data[data['type'] == type_list[i]]['ts_rmse'])
        box_list2.append(data[data['type'] == type_list[i]]['ts_mae'])

    plt.figure(figsize=(10, 5))  
    plt.subplot(1, 2, 1)
    plt.title('Test RMSE', fontsize=20)
    labels = type_list  
    plt.boxplot(box_list, labels=labels, vert=False, showmeans=True)  

    plt.subplot(1, 2, 2)
    plt.title('Test MAE', fontsize=20)
    labels = type_list  
    plt.boxplot(box_list2, labels=labels, vert=False, showmeans=True)

    plt.savefig(r'D:\projects\T-GCN-master\saved_model\box_plot.pdf')
    plt.show()  


def get_manifold_data(mode='HLLE', data_set='left', k=50, normalization='max_min', color='rgb',
                      shuffle=True, save_path=None, idx=None):
    file = "data/YC0{}_rel.csv".format(data_set)
    train_data = np.loadtxt(fname=file, delimiter=',', skiprows=1)[:, 0:3]
    if shuffle:
        np.random.shuffle(train_data)

    # choose color
    colors = get_color(data_set=data_set)

    # normalization
    for i in [0, 1, 2]:
        h = train_data[:, i]
        if normalization == 'max_min':
            temp = (h - h.min()) / (h.max() - h.min())
        elif normalization == 'mean_std':
            temp = (h - np.mean(h)) / np.std(h)
        elif normalization == 'nrom':
            temp = h / np.sum(h)
        else:
            temp = h
        train_data[:, i] = temp

    # manifold learning
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
    elif mode == 'UMAP':
        reducer = umap.UMAP(n_neighbors=k, n_components=2, metric='euclidean', random_state=42)
        reducer.fit(train_data)
        trans_data = reducer.transform(train_data)
    elif mode == 'TERME':
        trans_data = TERME(train_data, n_neighbors=k, n_components=2, gamma=1)
    else:
        trans_data = train_data

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = "{}_{}_{}_{}_{}_{}".format(data_set, mode, normalization, k, str(shuffle), idx)
        np.savetxt(save_path + "/{}.csv".format(file_name), trans_data, delimiter=',', fmt='%.64e')
        # show_manifold_simple(save_path + "/{}.csv".format(file_name), data_set=data_set, img_name=file_name)
    return trans_data


def show_manifold_simple(file, img_name, color='rgb', rotation=False, data_set='left', save_path=r'D:\projects\T-GCN-master\data\manifold_data\manifold_shape'):
    data = np.loadtxt(fname=file, delimiter=',')
    colors = get_color(color, data_set=data_set)

    if rotation:
        x = data.mean(axis=0)[0]
        y = data.mean(axis=0)[1]
        for i in range(data.shape[0]):
            data[i] = 2 * np.array([x, y]) - data[i]

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.scatter(data[:, 0], data[:, 1], marker='o', c=colors, s=20)

    plt.savefig(save_path + "\{}_{}.pdf".format(data_set, img_name), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    # plt.show()
    plt.cla()


def draw_three_d_surface(file="data/use_info_{}.csv".format('left'), color='rgb'):
    train_data = np.loadtxt(fname=color_file, delimiter=',', skiprows=1)[:, 0:3]
    data = train_data
    x = train_data[:, 0]
    y = train_data[:, 1]
    z = train_data[:, 2]

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()
    # best-fit quadratic curve (2nd-order)
    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2, data[:, :2] ** 3]
    C, _, _, _ = np.linalg.lstsq(A, z)
    # A * C = z
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2, XX ** 3, YY ** 3], C).reshape(X.shape)

    # choose color
    colors = get_color(color)

    # plot points and fitted surface using Matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.set_zlim(1500, 2600)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(300))

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.4)
    ax.scatter(x, y, z, s=2, c=colors, marker='o')
    plt.savefig("save_model/3d_origin_s.pdf", quality=100, dpi=500, bbox_inches='tight', transparent=True,
                pad_inches=0)
    plt.show()


def draw_three_d_scatter(dataset='2', color='rgb'):
    file = "data/YC0{}_rel.csv".format(dataset)
    train_data = np.loadtxt(fname=file, delimiter=',', skiprows=1)[:, 0:3]

    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    # plt.rc('font', family='serif')
    # plt.rcParams['font.sans-serif'] = ['Times new Roman']
    # plt.rcParams['font.serif'] = ['Helvetica']
    # plt.rcParams['font.sans-serif'] = ['Helvetica']

    x = train_data[:, 0]
    y = train_data[:, 1]
    z = train_data[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)

    colors = get_color(data_set=dataset)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(10000))
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.scatter(x, y, z, s=20, c=colors, marker='o')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # plt.legend()
    plt.savefig("results/3d/3d_origin_{}.pdf".format(dataset), quality=100, dpi=500, bbox_inches='tight', transparent=True,
                pad_inches=0)
    plt.show()


def draw_two_d_all_scatter(color='rgb'):
    # color = 'rgb'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), tight_layout=True)

    ax.set_xlabel('Longitude', size=35)
    ax.set_ylabel('Latitude', size=35)
    ax.set_ylim([30.65, 30.692])
    ax.set_xlim([1.02e2 + 0.03, 1.02e2 + 0.065])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False
    )

    dataset = 'left'
    file = "data/use_info_{}.csv".format(dataset)
    train_data = np.loadtxt(fname=color_file, delimiter=',', skiprows=1)[:, 0:3]
    x = train_data[:, 0]
    y = train_data[:, 1]
    z = train_data[:, 2]
    colors = get_color(data_set=dataset, color=color)

    dataset = 'right'
    file = "data/use_info_{}.csv".format(dataset)
    train_data = np.loadtxt(fname=color_file, delimiter=',', skiprows=1)[:, 0:3]
    # add margin
    x = np.concatenate((x, train_data[:, 0] + 0.001), axis=0)
    y = np.concatenate((y, train_data[:, 1]), axis=0)
    z = np.concatenate((z, train_data[:, 2]), axis=0)
    colors = np.concatenate((colors, get_color(data_set=dataset, color=color)), axis=0)

    # ax.scatter(x, y, z, s=20, c=colors, marker='o')
    plt.scatter(x, y, marker='o', c=colors, s=10)

    plt.savefig("saved_model/2d_origin.pdf", quality=100, dpi=500, bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
    plt.show()


def draw_two_d_scatter(dataset="2", color='rgb'):
    file = "data/YC0{}_rel.csv".format(dataset)
    train_data = np.loadtxt(fname=file, delimiter=',', skiprows=1)[:, 0:3]
    x = train_data[:, 0]
    y = train_data[:, 1]
    z = train_data[:, 2]

    # file = "data/YC0{}_rel.csv".format(1)
    # train_data = np.loadtxt(fname=file, delimiter=',', skiprows=1)[:z.shape[0], 0:3]
    # ly = train_data[:, 1]

    maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    for cm in maps:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), tight_layout=True)
        plt.axis('off')
        ax.scatter(x, y, s=15, c=z, marker='o', cmap=cm)
        plt.savefig("results/2d/2d_origin_{}_{}.pdf".format(dataset, cm), quality=100, dpi=500, bbox_inches='tight',
                    transparent=True,
                    pad_inches=0)
    # plt.show()


def draw_locally_adj(file='data/3left_TERMEnew_max_min_450_False_0.csv', k=4, local_ranges=None, color='rgb',
                     save_path='saved_model/local_adj', show_origin=False):
    if local_ranges is None:
        local_ranges = [[-0.02, -0.013, 0.01, 0], [0.005, 0.01, -0.0075, -0.02]]

    colors = get_color(color)

    data = np.loadtxt(fname=file, delimiter=',')
    if show_origin:
        plt.figure(figsize=(10, 10))
        plt.scatter(data[:, 0], data[:, 1], marker='o', c=colors, s=5)
        plt.savefig(save_path + "/origin_scale.pdf", quality=100, dpi=400, bbox_inches='tight',
                        transparent=True, pad_inches=0)
        plt.cla()
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.scatter(data[:, 0], data[:, 1], marker='o', c=colors, s=5)
        plt.savefig(save_path + "/origin.pdf", quality=100, dpi=400, bbox_inches='tight',
                        transparent=True, pad_inches=0)
        plt.cla()
        # add rectangle
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.scatter(data[:, 0], data[:, 1], marker='o', c=colors, s=5)
        now = plt.gca()
        local_ranges = [[-0.02, -0.013, 0.01, 0], [0.005, 0.01, -0.0075, -0.02]]
        for local_range in local_ranges:
            left, right, up, bottom = local_range
            rec = plt.Rectangle((left, bottom), width=right-left, height=up-bottom, edgecolor='r',
                                linewidth=1, facecolor='none')
            now.add_patch(rec)

        plt.savefig(save_path + "/origin_rectangle.pdf", quality=100, dpi=500, bbox_inches='tight',
                        transparent=True, pad_inches=0)
        plt.cla()

    # xy，rgb
    # return 1
    data_with_color = np.concatenate((data, colors), axis=1)

    for idx in range(len(local_ranges)):
        local_range = local_ranges[idx]
        left, right, up, bottom = local_range
        local_data = data_with_color[left < data_with_color[:, 0]]
        local_data = local_data[local_data[:, 0] < right]
        local_data = local_data[local_data[:, 1] < up]
        local_data = local_data[bottom < local_data[:, 1]]
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.scatter(local_data[:, 0], local_data[:, 1], marker='o', c=local_data[:, 2:], s=170)
        plt.savefig(save_path + "/local_nodes_{}_{}.pdf".format(k, idx), quality=100, dpi=400)
        # plt.show()
        plt.cla()

        # find neighbors
        X = local_data.copy()
        knn = NearestNeighbors(n_neighbors=k).fit(X)
        neighbors_ind = knn.kneighbors(X, return_distance=False)[:, 1:]
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        for i in range(neighbors_ind.shape[0]):
            for j in neighbors_ind[i]:
                plt.plot([X[i][0], X[j][0]], [X[i][1], X[j][1]], color=X[i][2:], alpha=0.2, linewidth=5)
        plt.scatter(local_data[:, 0], local_data[:, 1], marker='o', c=local_data[:, 2:], s=500)
        plt.savefig(save_path + "/local_adj_{}_{}.pdf".format(k, idx), quality=100, dpi=400, bbox_inches='tight',
                        transparent=True, pad_inches=0)


def draw_line(file="data/use_info_left.csv", representative=True):
    d = pd.read_csv(file, header=None, sep=',', index_col=False)
    d = np.loadtxt(fname=color_file, delimiter=',', skiprows=1)[:, 3:]

    fig = plt.figure(figsize=(20, 8), dpi=200)
    plt.xticks(np.linspace(1, 20, 20))
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    if not representative:
        # all data
        for i in range(10):
            k = i * 5
            for j in range(5):
                plt.plot(d[k + j, :], label=k + j)
            # plt.legend()
            plt.title('{}group'.format(i))
    else:
        # representative data
        a = [1, 6, 14, 17]
        b = [22, 25, 41, 48]
        c = [1, 6, 14, 17, 21, 22, 25, 41, 48]
        for j in b:
            plt.plot(d[j, :], label=j)
        # plt.legend()
        plt.ylabel('Deformation')
        plt.xlabel('Time')
    plt.savefig(r'D:\papers\AAAI2021-Landslides\pics\deformation_line_chart.pdf')
    # plt.show()


def draw_prediction(get_id_list=False, seq_len=3, pre_len=1):
    def recover_from_scatter(scatter_X, scatter_Y, seq_len):
        """
        Parameters
        ----------
        scatter_X shape of [_, seq_len, 4569]
        scatter_Y shape of [_, pre_len, 4569]

        Returns origin shape of [len(dataset) , 4569]
        -------

        """
        X = scatter_X[:, 0, :]
        Y = scatter_Y[:, 0, :]
        origin = X[0:seq_len]
        origin = np.concatenate((origin, Y), axis=0)
        return origin

    def get_X_and_Y(data, seq_len=3, pre_len=1):
        # same as load_data
        data = data.transpose()
        X, Y = [], []
        for i in range(len(data) - seq_len - pre_len + 1):
            X.append(data[i: i + seq_len])
            Y.append(data[i + seq_len: i + seq_len + pre_len])
        return np.array(X), np.array(Y)

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    file = "data/use_info_left.csv"
    origin_data = np.loadtxt(file)[:, 3:]
    num_nodes = origin_data.shape[0]

    file = "saved_model/Y_pre_saved/{}.csv".format("left_all_0.12041")
    pre = np.loadtxt(file, delimiter=',')
    pre = np.reshape(pre, [-1, pre_len, num_nodes])

    max_value = np.max(origin_data)
    pre = pre * max_value

    all_X, all_Y = get_X_and_Y(origin_data)
    pre_data = recover_from_scatter(all_X, pre, seq_len).transpose()


    if get_id_list:
        id_eva = pd.DataFrame(columns=['id', 'rmse'])
        for idx in range(pre_data.shape[0]):
            pre_one = pre_data[idx, :]
            origin_one = origin_data[idx, :]
            id_eva = id_eva.append({'id': idx, 'rmse': rmse(pre_one, origin_one)}, ignore_index=True)
        id_eva = id_eva.sort_values('rmse')
        id_list = np.array(id_eva[1700:1720]['id']).astype(np.int)
    else:
        id_list = [0, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045,
         3046, 3047, 3048, 3049, 3050, 3051, 3052, 3066, 3065]
        # [3993 1302  393 3363  750 3882 2557 1930   68 1285 2805 4284 3870 1233, 4473  479 1146  993  941 2361]
        best_list = [3037, 3039, 3045, 3038, 3044]

    # font setting
    legend_font = FontProperties()
    legend_font.set_size(27)
    label_font_s = 28
    tick_font_s = 23
    font = FontProperties()
    # font.set_family('serif')
    # font.set_name('Times New Roman')
    # font.set_style('italic')
    font.set_size(label_font_s)

    x_tic = np.arange(0, 21, 1)
    for i in range(20):
        fig = plt.figure(figsize=(15, 5))
        idx = id_list[i]
        pre_one = pre_data[idx, :]
        origin_one = origin_data[idx, :]
        #     plt.plot(x_tic[2:], pre_one[2:], linestyle='--',label='Prediction', c="#CC3333", linewidth=3)
        #     plt.vlines(2, pre_one[2]-2, pre_one[2]+2, colors = "#003366", linestyles = "dashed",label='Prediction Start')
        plt.plot(x_tic, pre_one, linestyle='--', label='Prediction', c="#CC3333", linewidth=3)
        plt.plot(x_tic, origin_one, linestyle='-', label='Truth', c="#000000")
        plt.xlabel('Time', fontproperties=font)
        plt.ylabel('Displacement(mm)', fontproperties=font)
        # plt.xticks((0, 2, 5, 10, 15, 20), fontsize=tick_font_s)
        plt.xticks([0, 5, 10, 15, 20], ['11/30/2018', '02/10/2019', '04/11/2019', '06/10/2019', '08/09/2019'],
                   fontsize=tick_font_s)
        plt.yticks((-5, 0, 5), fontsize=tick_font_s)
        plt.legend(prop=legend_font)
        plt.savefig('saved_model/pre_fig' + "/{}.pdf".format(idx), quality=100, dpi=500, bbox_inches='tight',
                    transparent=True, pad_inches=0)
        # plt.show()


if __name__ == '__main__':
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    dataset = 2
    # file = "data/YC0{}_rel.csv".format(dataset)
    # draw_two_d_scatter(dataset=dataset)
    draw_three_d_scatter(dataset='2')


    # show manifold shape
    a = ['MLLE', 'Isomap', 'T-SNE', 'UMAP']
    b = ['HLLE', 'LTSA']
    c = ['TERME']
    d = ['LLE', 'TERME']

    # data_set = '2'
    # save_path = r"D:\projects\SF_baselines\T-GCN-master\results\manifold"
    # for mode in c:
    #     get_manifold_data(data_set=data_set, mode=mode, k=1000, color='rgb', shuffle=False,
    #                                   save_path=save_path, idx=0)
    #     file_path = save_path + "\{}_{}_max_min_1000_False_0.csv".format(data_set, mode)
    #     show_manifold_simple(file_path, mode, data_set=data_set, save_path=save_path)


    # for mode in ['TERME']:
    #     # for m in ['max_min', 'mean_std', 'nrom', 'None']:
    #     for m in ['max_min']:
    #         # 5-60 -1. 60-100 -2
    #         for i in range(86, 100, 3):
    #             get_manifold_data(data_set='right', mode=mode, k=i, normalization=m, color='rgb', shuffle=False,
    #                               save_path=r"D:/projects/T-GCN-master/data/TERME", idx=0)
    #             print(i)
    #
    # get_manifold_data(data_set='right', mode='origin', k=2, normalization='max_min', color='rgb', shuffle=False,
    #                   save_path=r"D:/projects/T-GCN-master/data/manifold_data", idx=0)


    # for mode in ['UMAP']:
    #     file_path = r"D:\projects\T-GCN-master\data\manifold_data\left_{}_max_min_100_False_0.csv".format(mode)
    #     show_manifold_simple(file_path, mode)
    #
    # for idx in range(100, 700, 100):
    #     # file_path = r"D:\projects\T-GCN-master\data\manifold_data\left_{}_max_min_100_False_0.csv".format(mode)
    #     file_path = r"D:\projects\T-GCN-master\data\TERME\right_TERME_max_min_{}_False_0.csv".format(idx)
    #     show_manifold_simple(file_path, 'right_TERME_'+str(idx), data_set='right')

    # draw_two_d_all_scatter(color='rgb')
    # draw_three_d_scatter()



    # draw_three_d_surface()
    # draw_box_plot()
    # draw_line()
    # draw_locally_adj(local_ranges=[[-0.02, -0.01, 0.01, 0], [0.005, 0.01, 0, -0.02]], k=8, show_origin=True)
    # draw_prediction(get_id_list=True)
    # draw_locally_adj(show_origin=False, k=8, color='rgb')
