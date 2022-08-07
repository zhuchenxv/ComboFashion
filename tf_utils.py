from __future__ import division

import os

import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import random
import __init__

dtype = tf.float32 if __init__.config['dtype'] == 'float32' else tf.float64
minval = __init__.config['minval']
maxval = __init__.config['maxval']
mean = __init__.config['mean']
stddev = __init__.config['stddev']


def get_variable(init_type='xavier', shape=None, name=None, minval=minval, maxval=maxval, mean=mean,
                 stddev=stddev, dtype=dtype, ):
    if type(init_type) is str:
        init_type = init_type.lower()
    if init_type == 'tnormal':
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'uniform':
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'normal':
        return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'xavier':
        maxval = np.sqrt(6. / np.sum(shape))
        minval = -maxval
        print(name, 'initialized from:', minval, maxval, " shape:", shape)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'xavier_out':
        maxval = np.sqrt(3. / shape[1])
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'xavier_in':
        maxval = np.sqrt(3. / shape[0])
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'zero':
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=name)
    elif init_type == 'one':
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=name)
    elif init_type == 'identity' and len(shape) == 2 and shape[0] == shape[1]:
        return tf.Variable(tf.diag(tf.ones(shape=shape[0], dtype=dtype)), name=name)
    elif 'int' in init_type.__class__.__name__ or 'float' in init_type.__class__.__name__:
        return tf.Variable(tf.ones(shape=shape, dtype=dtype) * init_type, name=name)


def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def activate(weights, act_type):
    if type(act_type) is str:
        act_type = act_type.lower()
    if act_type == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif act_type == 'softmax':
        return tf.nn.softmax(weights)
    elif act_type == 'relu':
        return tf.nn.relu(weights)
    elif act_type == 'tanh':
        return tf.nn.tanh(weights)
    elif act_type == 'elu':
        return tf.nn.elu(weights)
    elif act_type == 'selu':
        return selu(weights)
    elif act_type == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo):
    opt_algo = opt_algo.lower()
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer
    elif opt_algo == 'moment':
        return tf.train.MomentumOptimizer
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer
    elif opt_algo == 'gd' or opt_algo == 'sgd':
        return tf.train.GradientDescentOptimizer
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer
    else:
        return tf.train.GradientDescentOptimizer


def get_loss(loss_func):
    loss_func = loss_func.lower()
    if loss_func == 'weight' or loss_func == 'weighted':
        return tf.nn.weighted_cross_entropy_with_logits
    elif loss_func == 'sigmoid':
        return tf.nn.sigmoid_cross_entropy_with_logits
    elif loss_func == 'softmax':
        return tf.nn.softmax_cross_entropy_with_logits


def check(x):
    try:
        return x is not None and x is not False and float(x) > 0
    except TypeError:
        return True


def get_l1_loss(params, variables):
    _loss = None
    with tf.name_scope('l1_loss'):
        for p, v in zip(params, variables):
            print('add l1', p, v)
            if not type(p) is list:
                if check(p):
                    if type(v) is list:
                        for _v in v:
                            if _loss is None:
                                _loss = tf.contrib.layers.l1_regularizer(p)(_v)# tf.nn.l1_loss(_v)
                            else:
                                _loss += tf.contrib.layers.l1_regularizer(p)(_v)
                    else:
                        if _loss is None:
                            _loss = tf.contrib.layers.l1_regularizer(p)(v)
                        else:
                            _loss += tf.contrib.layers.l1_regularizer(p)(v)
            else:
                for _lp, _lv in zip(p, v):
                    if _loss is None:
                        _loss = tf.contrib.layers.l1_regularizer(_lp)(_lv)
                    else:
                        _loss += tf.contrib.layers.l1_regularizer(_lp)(_lv)
    return _loss


def get_l2_loss(params, variables):
    _loss = None
    with tf.name_scope('l2_loss'):
        for p, v in zip(params, variables):
            print('add l2', p, v)
            if not type(p) is list:
                if check(p):
                    if type(v) is list:
                        for _v in v:
                            if _loss is None:
                                _loss = p * tf.nn.l2_loss(_v)
                            else:
                                _loss += p * tf.nn.l2_loss(_v)
                    else:
                        if _loss is None:
                            _loss = p * tf.nn.l2_loss(v)
                        else:
                            _loss += p * tf.nn.l2_loss(v)
            else:
                for _lp, _lv in zip(p, v):
                    if _loss is None:
                        _loss = _lp * tf.nn.l2_loss(_lv)
                    else:
                        _loss += _lp * tf.nn.l2_loss(_lv)
    return _loss


def normalize(norm, x, scale):
    if norm:
        return x * scale
    else:
        return x


def mul_noise(noisy, x, training=None):
    if check(noisy) and training is not None:
        with tf.name_scope('mul_noise'):
            noise = tf.truncated_normal(
                shape=tf.shape(x),
                mean=1.0, stddev=noisy)
            return tf.where(
                training,
                tf.multiply(x, noise),
                x)
    else:
        return x


def add_noise(noisy, x, training):
    if check(noisy):
        with tf.name_scope('add_noise'):
            noise = tf.truncated_normal(
                shape=tf.shape(x),
                mean=0, stddev=noisy)
            return tf.where(
                training,
                x + noise,
                x)
    else:
        return x


def drop_out(training, keep_probs, ):
    with tf.name_scope('drop_out'):
        keep_probs = tf.where(training,
                              keep_probs,
                              np.ones_like(keep_probs),
                              name='keep_prob')
    return keep_probs


def linear(xw):
    with tf.name_scope('linear'):
        l = tf.squeeze(tf.reduce_sum(xw, 1))
    return l


def output(x):
    with tf.name_scope('output'):
        if type(x) is list:
            logits = sum(x)
        else:
            logits = x
        outputs = tf.nn.sigmoid(logits)
    return logits, outputs


def layer_normalization(x, reduce_dim=1, out_dim=None, scale=None, bias=None):
    if type(reduce_dim) is int:
        reduce_dim = [reduce_dim]
    if type(out_dim) is int:
        out_dim = [out_dim]
    with tf.name_scope('layer_norm'):
        layer_mean, layer_var = tf.nn.moments(x, reduce_dim, keep_dims=True)
        x = (x - layer_mean) / tf.sqrt(layer_var)
        if scale is not False:
            scale = scale if scale is not None else tf.Variable(tf.ones(out_dim), dtype=dtype, name='g')
        if bias is not False:
            bias = bias if bias is not None else tf.Variable(tf.zeros(out_dim), dtype=dtype, name='b')
        if scale is not False and bias is not False:
            return x * scale + bias
        elif scale is not False:
            return x * scale
        elif bias is not False:
            return x + bias
        else:
            return x


def bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, node_in, batch_norm=False, layer_norm=False, training=True,
            res_conn=False):
    layer_kernels = []
    layer_biases = []
    x_prev = None
    for i in range(len(layer_sizes)):
        with tf.name_scope('hidden_%d' % i):
            wi = get_variable(init, name='w_%d' % i, shape=[node_in, layer_sizes[i]])
            bi = get_variable(0, name='b_%d' % i, shape=[layer_sizes[i]])
            print(wi.shape, bi.shape)
            print(layer_acts[i], layer_keeps[i])

            h = tf.matmul(h, wi)
            if i < len(layer_sizes) - 1:
                if batch_norm:
                    h = tf.layers.batch_normalization(h, training=training, name='mlp_bn_%d' % i)
                elif layer_norm:
                    h = layer_normalization(h, out_dim=layer_sizes[i], bias=False)
            h = h + bi
            if res_conn:
                if x_prev is None:
                    x_prev = h
                elif layer_sizes[i-1] == layer_sizes[i]:
                    h += x_prev
                    x_prev = h

            h = tf.nn.dropout(
                activate(
                    h, layer_acts[i]),
                layer_keeps[i])
            node_in = layer_sizes[i]
            layer_kernels.append(wi)
            layer_biases.append(bi)
    return h, layer_kernels, layer_biases


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    try:
        group_auc = float(total_auc) / impression_total
        # group_auc = round(group_auc, 4)
        return group_auc
    except:
        return None


def attention(queries, keys, keys_length, name, training, attention_size=[160, 80], reuse=False):
    '''
    queries:     [Batch, K]
    keys:        [Batch, history length, K]
    keys_length: [Batch]
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1]  # K
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # B, K * H
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # B, H, K
    din_all = tf.concat([queries, keys, queries-keys, queries*keys, queries+keys], axis=-1)  # B, H, 5K
    d_layer_1_all = tf.layers.dense(din_all, attention_size[0], activation=None, name='f1_att' + str(name), reuse=reuse)
    d_layer_1_all = tf.layers.batch_normalization(d_layer_1_all, training=training, name='bn_attention_f1'+str(name), reuse=reuse)
    d_layer_1_all = tf.nn.relu(d_layer_1_all)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, attention_size[1], activation=None, name='f2_att' + str(name), reuse=reuse)
    d_layer_2_all = tf.layers.batch_normalization(d_layer_2_all, training=training, name='bn_attention_f2' + str(name), reuse=reuse)
    d_layer_2_all = tf.nn.relu(d_layer_2_all)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att'+str(name), reuse=reuse)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B, 1, H
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, H]  set True according to key_length
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, H]
    # paddings = tf.ones_like(outputs) * (-float("inf"))#  (-2 ** 32 + 1)
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, H]
    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, H]
    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, K]
    return outputs







