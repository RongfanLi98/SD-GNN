# -*- coding: utf-8 -*-
import math
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import numpy.linalg as la
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score


def parse_argus():
    # parse the argus
    flags = tf.app.flags
    flags.DEFINE_string('f', '', 'kernel')
    flags.DEFINE_integer('training_epoch', 600, 'Number of epochs to train.')
    flags.DEFINE_float('train_rate', 0.5, 'rate of training set.')
    flags.DEFINE_float('val_rate', 0.3, 'rate of validation set.')
    flags.DEFINE_integer('seq_len', 5, '  time length of inputs.')
    flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
    flags.DEFINE_integer('batch_size', 16, 'batch size.')
    flags.DEFINE_integer('patient', 50, 'the max patient for early stop.')
    # learning rate control
    flags.DEFINE_integer('con', 0, 'continue training, 0 is not')
    flags.DEFINE_float('learning_rate', 0.0003, 'Initial learning rate, decay in training')
    flags.DEFINE_integer('decay_steps', 100, 'Initial learning rate.recommend, decay in model')
    flags.DEFINE_float('decay_rate', 0.9, 'Initial learning rate.recommend, decay in model')
    # main parameters
    flags.DEFINE_string('dataset', '1', 'the name of dataset: left or right or 1 or 2.')
    flags.DEFINE_string('stable_data', "data/manifold_data/{}/stable/YC0{}_rel_{}_max_min_300.csv".format("TERME", 1, "TERME"),
                        'choose stable data, avoiding manifold data is not stable. LLE UMAP. YC01 YC02')
    flags.DEFINE_string('model_name', 'tgcn', 'select a model: tgcn, lstm, gru')
    flags.DEFINE_integer('model_layers', 2, 'model_layers of gru and lstm.')  
    flags.DEFINE_integer('gru_units', 50, 'hidden units of tgcn gru.')
    flags.DEFINE_integer('threshold', 50, 'threshold of graph distance, above it will be seen as zero.')
    flags.DEFINE_string('adj_mode', 'mani_thres', 'plain, weight, manifold, mani_thres')
    flags.DEFINE_integer('self_w', 10, 'self weight')
    # log and save the results
    flags.DEFINE_string('key', "plain", 'the primary key of log')
    flags.DEFINE_integer('K', 300, 'near neighbors of KNN')
    flags.DEFINE_string('save_path', r"D:\projects\SF_baselines\T-GCN-master\saved_model\pure_result.csv", 'the save path of results')
    flags.DEFINE_integer('all_data', 0, 'all data or test set')
    return flags.FLAGS


def evaluation(labels, predicts, acc_threshold=0.015):
    a, b = labels, predicts
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    acc = a[np.abs(a - b) < np.abs(acc_threshold)]
    acc = np.size(acc) / np.size(a)
    r2 = r2_score(a, b)
    evs = explained_variance_score(y_true=a, y_pred=b)
    return rmse, mae, acc, r2, evs


def ab_distance(a_lat, a_lng, b_lat, b_lng):
    earth_radius = 6370.856  
    math_2pi = math.pi * 2
    pis_per_degree = math_2pi / 360  

    def lat_degree2km(dif_degree=.0001, radius=earth_radius):
        return radius * dif_degree * pis_per_degree

    def lat_km2degree(dis_km=111, radius=earth_radius):
        return dis_km / radius / pis_per_degree

    def lng_degree2km(dif_degree=.0001, center_lat=22):
        real_radius = earth_radius * math.cos(center_lat * pis_per_degree)
        return lat_degree2km(dif_degree, real_radius)

    def lng_km2degree(dis_km=1, center_lat=22):
        real_radius = earth_radius * math.cos(center_lat * pis_per_degree)
        return lat_km2degree(dis_km, real_radius)
    center_lat = .5 * a_lat + .5 * b_lat
    lat_dis = lat_degree2km(abs(a_lat - b_lat))
    lng_dis = lng_degree2km(abs(a_lng - b_lng), center_lat)
    return math.sqrt(lat_dis ** 2 + lng_dis ** 2)


def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=1):
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    return tf.constant(adj.toarray(), tf.float32)
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    # return sparse_to_tuple(adj)


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial, name=name)


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
