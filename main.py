# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from models import TGCN, recurrent_model
from utils import evaluation, parse_argus, ab_distance
from load_data import load_features_labels, get_weight_adj_with_threshold, \
    load_mani_adj, data_vis, load_mani_adj_with_threshold, load_basic_adj_with_threshold
import os
import pickle


time_start = time.time()
# hyper_parameters setting  #####################
FLAGS = parse_argus()
model_name, data_name = FLAGS.model_name, FLAGS.dataset
train_rate, val_rate = FLAGS.train_rate, FLAGS.val_rate
seq_len, pre_len = FLAGS.seq_len, FLAGS.pre_len
patient, training_epoch = FLAGS.patient, FLAGS.training_epoch
lr, decay_steps, decay_rate = FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate
threshold = FLAGS.threshold
stable_data = FLAGS.stable_data
model_layers = FLAGS.model_layers
K = FLAGS.K
result_type = FLAGS.key

pkl_name = "data/YC0{}_rel.pkl".format(data_name)
if not os.path.isfile(pkl_name):
    file = "data/YC0{}_rel.csv".format(data_name)
    g_data = np.loadtxt(fname=file, delimiter=',', skiprows=1)[:, 0:3]

    dis_arr = np.empty((g_data.shape[0], g_data.shape[0]), dtype=float)
    h_arr = np.empty((g_data.shape[0], g_data.shape[0]), dtype=float)
    for i in range(g_data.shape[0]):
        for j in range(g_data.shape[0]):
            dis = ab_distance(g_data[i][1], g_data[i][0], g_data[j][1], g_data[j][0])
            dis_arr[i][j] = np.sqrt(np.square(dis) + np.square(g_data[i][2] - g_data[j][2]))
            height = np.abs(g_data[i][2] - g_data[j][2])
            h_arr[i][j] = height
    f = open(pkl_name, 'wb')
    pickle.dump((dis_arr, h_arr), f)
    f.close()

# stable_data = "data/manifold_data/{}_{}_max_min_{}_False_0.csv".format(data_name, result_type, K)
# stable_data = "data/TERME/3{}_TERME_max_min_{}_False_0.csv".format(data_name, K)

if stable_data:
    hyper_parameters = "{}_{}_{}".format(threshold, FLAGS.gru_units, K)
else:
    hyper_parameters = "{}_{}_adj_mode{}_threshold{}_gru{}_seq{}_pre{}_{}".format(data_name, model_name, FLAGS.adj_mode,
                                                                              threshold, seq_len, pre_len,
                                                                              FLAGS.learning_rate, FLAGS.gru_units)

model_path = "saved_model/" + hyper_parameters + '/'

# load data ################################
if FLAGS.adj_mode == 'plain':
    adj = load_basic_adj_with_threshold(file="data/YC0{}_rel.pkl".format(data_name), threshold=threshold, self_w=FLAGS.self_w)
elif FLAGS.adj_mode == 'weight':
    adj = get_weight_adj_with_threshold(file="data/YC0{}_rel.pkl".format(data_name), threshold=threshold, norm=False)
elif FLAGS.adj_mode == 'manifold':
    adj = load_mani_adj(file='data/use_info_{}.csv'.format(data_name), mode='HLLE', k=100, stable_data=stable_data)
elif FLAGS.adj_mode == 'mani_thres':
    adj = load_mani_adj_with_threshold(file="data/YC0{}_rel.csv".format(data_name), mode='HLLE', k=100,
                                       dis_file="data/YC0{}_rel.pkl".format(data_name), threshold=threshold, stable_data=stable_data)
num_nodes, max_value, trainX, trainY, valX, valY, testX, testY, all_X, all_Y= load_features_labels(
    file="data/YC0{}_rel.csv".format(data_name), train_rate=train_rate, val_rate=val_rate,
    seq_len=seq_len, pre_len=pre_len, all=True)

print(all_X.shape)
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

if model_name == 'tgcn':
    y_pred = TGCN(inputs, adj, FLAGS.gru_units, pre_len)
    # output size = [seq_num * pre_len, num_nodes]
elif model_name == 'lstm':
    y_pred = recurrent_model(inputs, adj, FLAGS.gru_units, model_layers, pre_len, 'lstm')
    # output size = [pre_len, num_nodes]
elif model_name == 'gru':
    y_pred = recurrent_model(inputs, adj, FLAGS.gru_units, model_layers, pre_len, 'gru')

# optimizer define ##########################
lambda_loss = 0.002
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
train_loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)  # loss
train_error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))  # rmse
val_loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)  # loss
val_error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))  # rmse
global_step = tf.Variable(0, trainable=False)
lr = tf.compat.v1.train.exponential_decay(learning_rate=lr, global_step=global_step, decay_steps=decay_steps,
                                          decay_rate=decay_rate, staircase=False)
optimizer = tf.train.AdamOptimizer(lr).minimize(train_loss, global_step=global_step)

# Initialize session ########################
variables = tf.global_variables()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
sess.run(tf.global_variables_initializer())

# TensorBoard
# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log = False
if train_log:
    train_writer = tf.compat.v1.summary.FileWriter("train_logs/"+TIMESTAMP)  # add sess.graph to see graph
    val_writer = tf.compat.v1.summary.FileWriter("val_logs/"+TIMESTAMP)
#
tl = tf.compat.v1.summary.scalar('train_loss', train_loss)
vl = tf.compat.v1.summary.scalar('val_loss', val_loss)

# merged = tf.compat.v1.summary.merge_all()

saver = tf.compat.v1.train.Saver(tf.global_variables())

# suffle validation
np.random.seed(100)
np.random.shuffle(valX)
np.random.seed(100)
np.random.shuffle(valY)
# start training ######
count = 0
min_val_loss = 100000

continue_training = FLAGS.con
if continue_training > 0:
    saver.restore(sess, save_path="saved_model/tuned_model_{}/".format(continue_training))

for epoch in range(training_epoch):
    _, loss_tr, rmse_tr, summary = sess.run([optimizer, train_loss, train_error, tl], feed_dict={inputs: trainX, labels: trainY})
    if train_log:  train_writer.add_summary(summary, epoch)

    loss_val, rmse_val, summary = sess.run([val_loss, val_error, vl], feed_dict={inputs: valX, labels: valY})
    if train_log:  val_writer.add_summary(summary, epoch)

    res_tr = "Iter:{}, train_loss:{:.4}, train_rmse:{:.4}, val_loss:{:.4}, val_rmse:{:.4}"\
        .format(epoch, loss_tr, rmse_tr, loss_val, rmse_val)

    if epoch % 100==0:
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        save_file = FLAGS.save_path
        print("test set result" + str(testX.shape))
        test_pred = sess.run([y_pred], feed_dict={inputs: testX, labels: testY})
        test_pred = np.reshape(test_pred, [-1, num_nodes])
        rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(np.reshape(testY, [-1, num_nodes]), test_pred)
        rmse_ts = rmse_ts * 10
        mae_ts = mae_ts * 10
        with open(save_file, mode='a') as fin:
            result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{}".format(rmse_ts, mae_ts, acc_ts, r2_ts, var_ts,
                                                                               time_stamp,
                                                                               hyper_parameters, data_name, result_type)
            fin.write(result)
            fin.close()

    # early stop.
    if loss_val >= min_val_loss:
        count += 1
        print(res_tr)
        # if count == 1: saver.save(sess, model_path)
    else:
        count = 0
        min_val_loss = loss_val
        # set small learning rate, and set big saving threshold corespongdingly
        print(res_tr + "\t current best")
        # if epoch > 300 and epoch % 5 == 0: saver.save(sess, model_path)
    if count >= patient and epoch > 50:
        print("Epoch : {} \t That's too bad, we should stop the training process...".format(epoch))
        break

# if epoch == training_epoch-1: saver.save(sess, model_path)

if train_log:
    train_writer.close()
    val_writer.close()
# saver.save(sess, model_path)

# show the test information ######
# print("restore model from: {}\n".format(model_path))
# saver.restore(sess, model_path)
if FLAGS.all_data == 1:
    # all
    print("all data set result")
    pre_data = sess.run([y_pred], feed_dict={inputs: all_X, labels: all_Y})
    pre_data = np.reshape(pre_data, [-1, num_nodes])
    rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(np.reshape(all_Y, [-1, num_nodes]), pre_data)
else:
    print("test set result"+str(testX.shape))
    test_pred = sess.run([y_pred], feed_dict={inputs: testX, labels: testY})
    test_pred = np.reshape(test_pred, [-1, num_nodes])
    rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(np.reshape(testY, [-1, num_nodes]), test_pred)

# note, the data is normed data, see function evaluation
# predict_ture_value = test_pred * max_value

# output
total_time = time.time() - time_start
print("\nTotal time: {:.2f}s".format(total_time))

time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

save_file = FLAGS.save_path

# scale is smaller than dam
rmse_ts = rmse_ts * 10
mae_ts = mae_ts * 10

with open(save_file, mode='a') as fin:
    result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{}".format(rmse_ts, mae_ts, acc_ts, r2_ts, var_ts,time_stamp,
                                                                       hyper_parameters, data_name, result_type)
    fin.write(result)
