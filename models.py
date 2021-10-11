import tensorflow as tf
from tgcn import tgcnCell
from tensorflow.keras import layers


def TGCN(_X, adj, gru_units, pre_len):
    """
    _X:         features  shape = (None, seq_len, num_nodes * 1), None is number of seq, which is seq_len long
                and every node is represented by other num_nodes nodes, through the adj
    adj:        the adj of graph
    gru_units:  the number of gru units, every rnn outputs shape
    pre_len:    the len of prediction
    """
    # Graph weights
    _weights = {'out': tf.Variable(tf.random.normal([gru_units, pre_len], mean=1.0), name='weight_o')}
    _biases = {'out': tf.Variable(tf.random.normal([pre_len]), name='bias_o')}
    num_nodes = adj.shape[0]

    # add muti rnn cell to improve performance
    # cell_1 = tf.compat.v1.nn.rnn_cell.LSTMCell(int(num_nodes/2))
    # cell_1 = tf.compat.v1.nn.rnn_cell.LSTMCell(num_nodes)
    cell_2 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_2], state_is_tuple=True) 

    _X = tf.unstack(_X, axis=1)  # seq_len length list of tensor, of which each tensor' shape is (seq_sum, num_nodes)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)

    # shape=(seq_num * num_nodes, gru_units)
    o = tf.reshape(outputs[-1], shape=[-1, num_nodes, gru_units])
    last_output = tf.reshape(o, shape=[-1, gru_units])

    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    # output size = [seq_num * pre_len, num_nodes]
    return output


def recurrent_model(inputs, adj, units, model_layers: int, pre_len, cell_type='lstm'):
    """
    Args:
        inputs:     features  shape = (seq_num, seq_len, embedding_dim), None is number of seq, which is seq_len long
                    train data is shape of (7, 3, 4569)
        units:      Positive integer, dimensionality of the output space.
    Returns:
    """
    num_nodes = adj.shape[0]
    seq_num = inputs.shape[0]
    seq_len = inputs.shape[1]

    if cell_type == 'lstm':
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=units) for _ in range(model_layers)])
    elif cell_type == 'gru':
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_units=units) for _ in range(model_layers)])

    x = tf.transpose(inputs, perm=[0, 2, 1])
    x = tf.reshape(x, [-1, seq_len, 1])
    # seq_len length list of tensor, of which each tensor' shape is (None, num_nodes)
    x = tf.unstack(x, axis=1)

    # outputs shape = (seq_len, num_nodes * seq_num, 1 * units)
    outputs, states = tf.nn.static_rnn(cell, x, dtype=tf.float32)

    output = outputs[-1]  # output shape = (num_nodes * seq_num, units)

    _weights = {'out': tf.Variable(tf.random.normal([units, pre_len], mean=1.0), name='weight_o')}
    _biases = {'out': tf.Variable(tf.random.normal([pre_len]), name='bias_o')}

    output = tf.matmul(output, _weights['out']) + _biases['out']  # output shape = (num_nodes * seq_num, pre_len)
    output = tf.reshape(output, [num_nodes, -1, pre_len])
    output = tf.transpose(output, perm=[2, 1, 0])                 # output shape = (pre_len, seq_num, num_nodes)
    output = tf.reshape(output, [-1, num_nodes])
    # output size = [seq_num * pre_len, num_nodes] keep the same as TGCN
    return output
