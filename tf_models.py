from __future__ import print_function

from abc import abstractmethod
import tensorflow as tf



import __init__
from tf_utils import drop_out, linear, output, get_l1_loss, bin_mlp, get_variable, get_l2_loss, attention

dtype = __init__.config['dtype']

if dtype.lower() == 'float32' or dtype.lower() == 'float':
    dtype = tf.float32
elif dtype.lower() == 'float64':
    dtype = tf.float64


class Model:
    inputs = None
    outputs = None
    logits = None
    labels = None
    learning_rate = None
    loss = None
    l1_loss = None
    l2_loss = None
    optimizer = None
    grad = None

    @abstractmethod
    def compile(self, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__


class ComboFashion(Model):
    def __init__(self, init='xavier', input_dim=None, multi_hot_input_dim=None, embed_size=None,
                 l2_w=None, l2_v=None, multi_hot_embed_size=None,
                 layer_sizes=None, multi_hot_length=0, multi_hot_field_num=0, one_hot_field_num=0,
                 layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, his_len=50,
                 history_embed_size=64, use_his_embs=True, use_his_visual=True,
                 embs_add_neg=False, visual_add_neg=False):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.l1_v = l1_v
        self.layer_l1 = layer_l1
        self.multi_hot_length = multi_hot_length
        self.multi_hot_field_num = multi_hot_field_num
        item_multi_hot_field_num = int(multi_hot_field_num/2)
        with tf.name_scope('input'):
            self.one_hot_inputs = tf.placeholder(tf.int32, [None, one_hot_field_num], name='one_hot_inputs')
            self.multi_hot_inputs = tf.placeholder(tf.int32, [None, multi_hot_field_num, multi_hot_length], name='multi_hot_inputs')
            self.multi_hot_feat_mask = tf.placeholder(tf.float32, [None, multi_hot_field_num, multi_hot_length], name='multi_hot_feat_mask')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.current_main_embs = tf.placeholder(tf.float32, [None, history_embed_size], name='current_main_embs')
            self.current_slide_embs = tf.placeholder(tf.float32, [None, history_embed_size], name='current_slide_embs')
            self.pos_main_his_embs = tf.placeholder(tf.float32, [None, his_len, history_embed_size], name='pos_main_his_embs')
            self.pos_main_his_embs_num = tf.placeholder(tf.int32, [None], name='pos_main_his_embs_num')
            self.pos_slide_his_embs = tf.placeholder(tf.float32, [None, his_len, history_embed_size], name='pos_slide_his_embs')
            self.pos_slide_his_embs_num = tf.placeholder(tf.int32, [None], name='pos_slide_his_embs_num')
            self.pos_main_his_visual = tf.placeholder(tf.int32, [None, his_len, item_multi_hot_field_num, multi_hot_length], name='pos_main_his_visual')
            self.pos_main_his_visual_mask = tf.placeholder(tf.float32, [None, his_len, item_multi_hot_field_num, multi_hot_length], name='pos_main_his_visual_mask')
            self.pos_main_his_visual_num = tf.placeholder(tf.int32, [None], name='pos_main_his_visual_num')
            self.pos_slide_his_visual = tf.placeholder(tf.int32, [None, his_len, item_multi_hot_field_num, multi_hot_length], name='pos_slide_his_visual')
            self.pos_slide_his_visual_mask = tf.placeholder(tf.float32, [None, his_len, item_multi_hot_field_num, multi_hot_length], name='pos_slide_his_visual_mask')
            self.pos_slide_his_visual_num = tf.placeholder(tf.int32, [None], name='pos_slide_his_visual_num')

        layer_keeps = drop_out(self.training, layer_keeps)
        w = get_variable(init, name='w', shape=[input_dim])
        w2 = get_variable(init, name='w2', shape=[multi_hot_input_dim])
        v = get_variable(init, name='v', shape=[input_dim, embed_size])
        v2 = get_variable(init, name='v2', shape=[multi_hot_input_dim, multi_hot_embed_size])
        b = get_variable('zero', name='b', shape=[1])
        xw1 = tf.gather(w, self.one_hot_inputs)
        xw2 = tf.reduce_sum(tf.gather(w2, self.multi_hot_inputs) * self.multi_hot_feat_mask, axis=-1) / tf.reduce_sum(self.multi_hot_feat_mask, axis=-1)
        self.xw = tf.concat([xw1, xw2], axis=-1)
        xv1 = tf.gather(v, self.one_hot_inputs)
        xv2 = tf.reduce_sum(tf.transpose(tf.transpose(tf.gather(v2, self.multi_hot_inputs), perm=[3, 0, 1, 2]) * self.multi_hot_feat_mask,perm=[1, 2, 0, 3]), axis=-1)
        xv1_ = tf.reshape(xv1, [-1, one_hot_field_num * embed_size])
        xv2_ = tf.reshape(xv2, [-1, multi_hot_field_num * multi_hot_embed_size])
        self.xv = tf.concat([xv1_, xv2_], axis=1)
        dnn_input_dim = one_hot_field_num * embed_size + multi_hot_field_num * multi_hot_embed_size
        h = self.xv

        his_main_pos_embs = tf.squeeze(
            attention(self.current_main_embs, self.pos_main_his_embs, self.pos_main_his_embs_num,
                        'pos_main_embs_attention', self.training), 1)
        his_slide_pos_embs = tf.squeeze(
            attention(self.current_slide_embs, self.pos_slide_his_embs, self.pos_slide_his_embs_num,
                        'pos_slide_embs_attention', self.training), 1)

        his_embs_pos_info = tf.concat(
            [self.current_main_embs, his_main_pos_embs, tf.multiply(self.current_main_embs, his_main_pos_embs),
                self.current_slide_embs, his_slide_pos_embs, tf.multiply(self.current_slide_embs, his_slide_pos_embs)],
            axis=-1)
        h = tf.concat([h, his_embs_pos_info], axis=-1)
        dnn_input_dim += history_embed_size * 6

        current_main_visual_xv = tf.reshape(tf.gather(xv2, [0, 2, 4], axis=1), (-1, item_multi_hot_field_num * multi_hot_embed_size))
        current_slide_visual_xv = tf.reshape(tf.gather(xv2, [1, 3, 5], axis=1), (-1, item_multi_hot_field_num * multi_hot_embed_size))
        pos_visual_main_xv = tf.reshape(tf.reduce_sum(tf.transpose(
            tf.transpose(tf.gather(v2, self.pos_main_his_visual),
                            perm=[4, 0, 1, 2, 3]) * self.pos_main_his_visual_mask,
            perm=[1, 2, 3, 0, 4]), axis=-1), (-1, his_len, item_multi_hot_field_num * multi_hot_embed_size))
        pos_visual_slide_xv = tf.reshape(tf.reduce_sum(tf.transpose(
            tf.transpose(tf.gather(v2, self.pos_slide_his_visual),
                            perm=[4, 0, 1, 2, 3]) * self.pos_slide_his_visual_mask,
            perm=[1, 2, 3, 0, 4]), axis=-1), (-1, his_len, item_multi_hot_field_num * multi_hot_embed_size))
        his_main_pos_visual = tf.squeeze(
            attention(current_main_visual_xv, pos_visual_main_xv, self.pos_main_his_visual_num,
                        'pos_main_visual_attention',
                        self.training), 1)
        his_slide_pos_visual = tf.squeeze(
            attention(current_slide_visual_xv, pos_visual_slide_xv, self.pos_slide_his_visual_num,
                        'pos_slide_visual_attention',
                        self.training), 1)
        his_visual_pos_info = tf.concat(
            [his_main_pos_visual, tf.multiply(current_main_visual_xv, his_main_pos_visual),
                his_slide_pos_visual, tf.multiply(current_slide_visual_xv, his_slide_pos_visual)],
            axis=-1)
        h = tf.concat([h, his_visual_pos_info], axis=-1)
        dnn_input_dim += multi_hot_embed_size * 4 * item_multi_hot_field_num

        l = linear(self.xw)
        h, self.layer_kernels, self.layer_biases = bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, dnn_input_dim,
                                                           batch_norm=batch_norm, layer_norm=layer_norm,
                                                           training=self.training)

        h = tf.squeeze(h)
        output_w1 = get_variable(init, name='output_w1', shape=[1])
        output_w2 = get_variable(init, name='output_w2', shape=[1])
        self.logits, self.outputs = output([output_w1 * l, output_w2 * h, b])

    def compile(self, loss=None, optimizer=None, global_step=None,
                pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    loss(logits=self.logits, targets=self.labels,
                         pos_weight=pos_weight))
                _loss_ = self.loss
                self.l1_loss = get_l1_loss(
                    [self.l1_w, self.l1_v, self.layer_l1, self.layer_l1],
                    [self.xw, self.xv, self.layer_kernels, self.layer_biases])
                if self.l1_loss is not None:
                    _loss_ += self.l1_loss
                self.l2_loss = get_l2_loss(
                    [self.l2_w, self.l2_v, self.layer_l2, self.layer_l2],
                    [self.xw, self.xv, self.layer_kernels, self.layer_biases])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                self.optimizer = optimizer.minimize(loss=_loss_,
                                                    global_step=global_step)



