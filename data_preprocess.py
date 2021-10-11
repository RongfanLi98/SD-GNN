# -*- coding:utf-8 -*-
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt


def get_useful_info():
    # read raw data sequence.
    sequence_data = list()
    out_the_border_right = [195305, 195036, 195035, 176341, 176343, 176345, 176691,
                            150510, 155324, 154829, 163069, 162157,
                            198089, 197066, 198542, 198543, 198316, 198794, 199042, 199041, 199040, 199039, 199038,
                            199037, 205874, 205877, 205875, 205878, 205879, 205678, 205681, 205679, 205682, 205680,
                            205683, 205475, 198532, 198309]
    with open(file="3d_sequence.txt", mode='r') as fin:
        for line in fin.readlines():
            row = [float(k) for k in line.split(',')]
            if row[0] in out_the_border_right:
                continue
            else:
                sequence_data.append(row[3:6] + row[10:])  # [lon, lat, height, D_1, D_2, ... , D_21]

    sequence_array = np.array(sequence_data)
    np.savetxt(fname="use_info.csv", X=sequence_array)


def split_data_in_two_parts():
    # 使用线性规划来将河道左右的点分割开来
    x1, y1 = 102.048089, 30.657121
    x2, y2 = 102.042647, 30.672335
    x3, y3 = 102.054991, 30.686387

    k_1 = (y2 - y1) / (x2 - x1)
    k_2 = (y3 - y2) / (x3 - x2)

    # line1: y - y2 + k_1 * (x2 - x) == 0
    # line2: y - y2 + k_2 * (x2 - x) == 0

    left_side, right_side = [], []
    use_info = np.loadtxt(fname="use_info.csv")
    for row in use_info:
        x, y = row[0], row[1]
        if y - y2 + k_1 * (x2 - x) > 0 and y - y2 + k_2 * (x2 - x) < 0:
            right_side.append(row)
        else:
            left_side.append(row)
    print("number of left:", len(left_side))
    print("number of right:", len(right_side))

    left_side = np.array(left_side, dtype=float)
    np.savetxt(fname="use_info_left.csv", X=left_side)
    right_side = np.array(right_side)
    np.savetxt(fname="use_info_right.csv", X=right_side)
    print("save done.")


def get_dis_by_loc(file="use_info.csv", save_file='raw_dis.pkl'):
    import haversine as hs
    points_loc = np.loadtxt(fname=file)[:, 0:2]
    points_heights = np.loadtxt(fname=file)[:, 2:3]  # height of every point.
    points_num = len(points_loc)
    dis_array = np.zeros((points_num, points_num), dtype=float)
    height_array = np.zeros((points_num, points_num), dtype=float)
    for i in range(0, points_num):
        for j in range(i, points_num):
            dis = round(hs.haversine(points_loc[i], points_loc[j], unit='m'), 2)
            dis_array[i][j] = dis
            height_array[i][j] = abs(points_heights[i] - points_heights[j])

    dis_array += dis_array.T  # 对角线元素依旧为0
    height_array += height_array.T
    with open(save_file, mode='wb') as f:
        pickle.dump(obj=(dis_array, height_array), file=f)
    print("save done. save_file:", save_file)


def plot_whisker(file_1='edge_whisker.pkl', file_2='edge_whisker.pkl'):
    # 绘制“边长”分布的箱线图
    import matplotlib.pyplot as plt
    with open(file_1, 'rb') as f:
        points_1 = pickle.load(file=f)
    with open(file_2, 'rb') as f:
        points_2 = pickle.load(file=f)

    plt.boxplot((points_1, points_2), labels=('Dataset_1', 'Dataset_2'))
    plt.show()


def plot_CDF(file_dis_1='raw_dis_left.pkl', file_dis_2='raw_dis_right.pkl'):
    # 绘制两个数据集“边长”分布的 CDF 累积分布函数

    def get_cdf_of_edges_dis(file_dis='raw_dis.pkl'):
        """
        统计并返回所有的边长，为研究“边长”的分布做准备
        """
        with open(file_dis, 'rb') as f:
            dis_arr, hei_arr = pickle.load(file=f)
        dis_arr = np.tril(dis_arr)
        distance = []
        for row in dis_arr:
            for item in row:
                if item != 0:
                    distance.append(item)
        print("File: {}, number of edge : {}".format(file_dis, len(distance)))
        diatance = sorted(distance)
        cumsum = np.cumsum(sorted(distance))
        cdf = list(map(lambda x: x / cumsum[-1], cumsum))
        return diatance, cdf

    points_1, cdf_1 = get_cdf_of_edges_dis(file_dis_1)
    points_2, cdf_2 = get_cdf_of_edges_dis(file_dis_2)

    # Plot both

    plt.plot(points_1, cdf_1, 'r--')
    plt.plot(points_2, cdf_2, 'b')

    plt.show()


def get_adj_with_threshold(file="raw_dis.pkl", threshold=100):
    with open(file, 'rb') as f:
        dis_arr, height_arr = pickle.load(file=f)

    # import matplotlib.pyplot as plt
    # # Plot a normalized histogram with 50 bins
    # plt.hist(dis_array, bins=50, density=1)  # matplotlib version (plot)
    # plt.savefig('raw_distance.png')
    # plt.show()

    dis_arr[dis_arr <= threshold] = 1
    dis_arr[dis_arr > threshold] = 0
    # dis_array = dis_array - np.eye(dis_array.shape[0])

    return dis_arr


def judge_graph_connectivity(adj):
    adj += np.eye(N=adj.shape[0])  # add self-loop
    edge_list = np.where(adj != 0)
    edge_list = np.array([edge_list[0], edge_list[1]]).transpose()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    visited = [0] * G.number_of_nodes()
    # visited = DFS(G, 0, visited)      // max iter of model.
    visited = BFS(G, 0, visited)
    return (0, G.number_of_edges()) if 0 in visited else (1, G.number_of_edges())


def DFS(G, node, visit):
    visit[node] = 1
    for nei in list(G.neighbors(node)):
        if visit[nei] == 0:
            visit = DFS(G, nei, visit)
    return visit


def BFS(G, node, visit):
    visit[node] = 1
    que = list()
    que.append(node)
    while len(que) != 0:
        node = que.pop()
        for nei in list(G.neighbors(node)):
            if visit[nei] == 0:
                visit[nei] = 1
                que.append(nei)
    return visit


def edge_with_threshold(save_file='att_with_thresh.pkl'):
    file_1 = 'raw_dis_left.pkl'
    file_2 = 'raw_dis_right.pkl'

    def get_attribute(file=''):
        attribute = []
        for i in range(10, 1501, 10):
            adj = get_adj_with_threshold(file=file, threshold=i)
            flag, num_edge = judge_graph_connectivity(adj)
            attribute.append([i, num_edge, flag])
            if i % 100 == 0:
                print("threshold: ", i)
        return attribute

    att_left = get_attribute(file_1)
    att_right = get_attribute(file_2)

    with open(save_file, mode='wb') as f:
        pickle.dump(obj=(att_left, att_right), file=f)
    print("save done. save_file:", save_file)


def plot_edge_attri(file='att_with_thresh.pkl'):
    with open(file, mode='rb') as f:
        att_left, att_right = pickle.load(f)

    def get_flag(arr):
        index, val = [], []
        for i, flag in enumerate(arr[2]):
            if flag == 1:
                index.append(arr[0][i])
                val.append(arr[1][i])
        return index, val

    # Plot both
    left_arr = np.array(att_left).transpose()
    right_arr = np.array(att_right).transpose()
    index_l, flag_l = get_flag(left_arr)
    index_r, flag_r = get_flag(right_arr)

    plt.plot(index_l, flag_l, c='y', alpha=0.4, linewidth=10)
    plt.plot(index_r, flag_r, c='y', alpha=0.4, linewidth=10)

    plt.plot([0] + left_arr[0], [0] + left_arr[1], 'r--', label='left')
    plt.plot([0] + right_arr[0], [0] + right_arr[1], 'b', label='right')

    # plt.scatter(index_l, flag_l, marker='o', c='y', alpha=0.8, s=20)
    # plt.scatter(index_r, flag_r, marker='8', c='r', alpha=0.8, s=20)

    plt.legend()
    plt.show()


# 绘制 graph 的点和边，图的可视化，没什么用，由于 points 太多了，画出来乱作一遭
# def build_graph_and_show():
#     adj = get_adj_with_threshold()
#     edge_list = np.where(adj == 1)
#     edge_list = np.array([edge_list[0], edge_list[1]]).transpose()
#     G = nx.Graph()
#     G.add_edges_from(edge_list)
#
#     import matplotlib.pyplot as plt
#     nx.draw(G, with_labels=True, font_weight='bold')
#     plt.savefig("raw_graph.png")
#     plt.show()


#  re-preprocess.  Add the height as weight information.
# have done the following ...
# get_useful_info()
# split_data_in_two_parts()
# get_dis_by_loc(file="use_info_left.csv", save_file='raw_dis_left.pkl')
# get_dis_by_loc(file="use_info_right.csv", save_file='raw_dis_right.pkl')
# plot_CDF(file_dis_1='raw_dis_left.pkl', file_dis_2='raw_dis_right.pkl')

# edge_with_threshold(save_file='att_with_thresh.pkl')
# plot_edge_attri(file='att_with_thresh.pkl')


if __name__ == "__main__":
    # get_useful_info()
    # split_data_in_two_parts()

    # get_dis_by_loc(file="use_info_left.csv", save_file='raw_dis_left.pkl')
    # get_dis_by_loc(file="use_info_right.csv", save_file='raw_dis_right.pkl')

    plot_edge_attri(file='att_with_thresh.pkl')
