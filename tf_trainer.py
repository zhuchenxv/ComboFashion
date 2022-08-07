

from __future__ import division
from __future__ import print_function

import datetime
import os
import time
import _pickle as cPickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss
from tf_utils import get_optimizer, get_loss, cal_group_auc
import __init__
import json


class Trainer:
    logdir = None
    session = None
    dataset = None
    model = None
    saver = None
    learning_rate = None
    train_pos_ratio = None
    test_pos_ratio = None

    def __init__(self, model=None, train_gen=None, test_gen=None, valid_gen=None, opt='adam', epsilon=1e-8,
                 initial_accumulator_value=1e-8, momentum=0.95, loss='weighted', pos_weight=1.0,
                 n_epoch=1, train_per_epoch=10000, test_per_epoch=10000, early_stop_epoch=5, batch_size=2000,
                 learning_rate=1e-2, decay_rate=0.95, test_every_epoch=1,
                 one_hot_pre_sum=None, multi_hot_pre_sum=None, his_len=50,
                 use_his_embs=True, use_his_visual=True,
                 embs_add_neg=False, visual_add_neg=False,):
        self.model = model
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.valid_gen = valid_gen
        optimizer = get_optimizer(opt)
        loss = get_loss(loss)
        self.pos_weight = pos_weight
        self.n_epoch = n_epoch
        self.train_per_epoch = train_per_epoch + 1
        self.early_stop_epoch = early_stop_epoch
        self.test_per_epoch = test_per_epoch
        self.batch_size = batch_size
        self._learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.call_auc = roc_auc_score
        self.call_loss = log_loss
        self.test_every_epoch = test_every_epoch
        self.multi_hot_length = self.model.multi_hot_length
        self.multi_hot_field_num = self.model.multi_hot_field_num
        self.his_len = his_len
        self.embs_add_neg = embs_add_neg
        self.visual_add_neg = visual_add_neg
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.one_hot_pre_sum = np.array(one_hot_pre_sum)
        self.multi_hot_pre_sum = np.array(multi_hot_pre_sum)
        self.multi_hot_pre_sum_main = np.array(self.multi_hot_pre_sum)[[0, 2, 4]]
        self.multi_hot_pre_sum_slide = np.array(self.multi_hot_pre_sum)[[1, 3, 5]]
        self.learning_rate = tf.placeholder("float")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.use_his_embs = use_his_embs
        self.use_his_visual = use_his_visual
        data_path = __init__.config['data_path']
        self.item_embs_dict = cPickle.load(open(data_path+"/item_embs.pkl", "rb"))
        self.item_embs_set = set(self.item_embs_dict.keys())
        self.item_id_to_actual_item_id = np.load(data_path+'/item_id_to_string_dict.npy').item()
        self.visual_dict = cPickle.load(open(data_path+"/visual_dict.pkl", "rb"))
        self.visual_set = set(self.visual_dict.keys())

        self.main_dict = cPickle.load(open(data_path+"/handler_dict/detail/main_dict.pkl", "rb"))
        self.slide_dict = cPickle.load(open(data_path+"/handler_dict/detail/slide_dict.pkl", "rb"))

        self.main_dict_set = {}
        for key in self.main_dict:
            self.main_dict_set[key] = set(self.main_dict[key].keys())
        self.slide_dict_set = {}
        for key in self.slide_dict:
            self.slide_dict_set[key] = set(self.slide_dict[key].keys())

        tf.summary.scalar('global_step', self.global_step)
        if opt == 'adam':
            opt = optimizer(learning_rate=self.learning_rate,
                            epsilon=self.epsilon)  # TODO fbh
        elif opt == 'adagrad':
            opt = optimizer(learning_rate=self.learning_rate,
                            initial_accumulator_value=initial_accumulator_value)
        elif opt == 'moment':
            opt = optimizer(learning_rate=self.learning_rate,
                            momentum=momentum)
        else:
            opt = optimizer(learning_rate=self.learning_rate, )  # TODO fbh
        self.model.compile(loss=loss, optimizer=opt,
                           global_step=self.global_step, pos_weight=pos_weight)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

    def _run(self, fetches, feed_dict):
        return self.session.run(fetches=fetches, feed_dict=feed_dict)

    def _train(self, one_hot_x, multi_hot_x, y, pos_his_embs_info, pos_his_visual_info):
        one_hot_x += self.one_hot_pre_sum
        feed_dict = {self.model.labels: y,
                     self.learning_rate: self._learning_rate,
                     self.model.one_hot_inputs: one_hot_x}

        feed_dict[self.model.pos_main_his_embs] = pos_his_embs_info[2]
        feed_dict[self.model.pos_main_his_embs_num] = pos_his_embs_info[3]
        feed_dict[self.model.pos_slide_his_embs] = pos_his_embs_info[4]
        feed_dict[self.model.pos_slide_his_embs_num] = pos_his_embs_info[5]
        feed_dict[self.model.current_main_embs] = pos_his_embs_info[0]
        feed_dict[self.model.current_slide_embs] = pos_his_embs_info[1]

        pos_main_his_visual = pos_his_visual_info[0]
        pos_main_his_visual_mask = np.where(pos_main_his_visual >= 0, 1, 0)
        for i in range(int(self.multi_hot_field_num / 2)):
            pos_main_his_visual[:, :, i, :] += self.multi_hot_pre_sum_main[i] * pos_main_his_visual_mask[:, :, i, :]
        feed_dict[self.model.pos_main_his_visual] = pos_main_his_visual
        feed_dict[self.model.pos_main_his_visual_mask] = pos_main_his_visual_mask
        feed_dict[self.model.pos_main_his_visual_num] = pos_his_visual_info[1]
        pos_slide_his_visual = pos_his_visual_info[2]
        pos_slide_his_visual_mask = np.where(pos_slide_his_visual >= 0, 1, 0)
        for i in range(int(self.multi_hot_field_num / 2)):
            pos_slide_his_visual[:, :, i, :] += self.multi_hot_pre_sum_slide[i] * pos_slide_his_visual_mask[:, :, i, :]
        feed_dict[self.model.pos_slide_his_visual] = pos_slide_his_visual
        feed_dict[self.model.pos_slide_his_visual_mask] = pos_slide_his_visual_mask
        feed_dict[self.model.pos_slide_his_visual_num] = pos_his_visual_info[3]

        if multi_hot_x.shape[1] > 0:
            multi_hot_x_mask = np.where(multi_hot_x >= 0, 1, 0)
            for i in range(self.multi_hot_field_num):
                multi_hot_x[:, i, :] += self.multi_hot_pre_sum[i] * multi_hot_x_mask[:, i, :]
            feed_dict[self.model.multi_hot_inputs] = multi_hot_x
            feed_dict[self.model.multi_hot_feat_mask] = multi_hot_x_mask
        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = True
        if self.model.l1_loss is None:
            _, _loss, outputs = self._run(
                fetches=[self.model.optimizer, self.model.loss,
                         self.model.outputs],
                feed_dict=feed_dict)
            _l1_loss = 0
        else:
            _, _loss, _l1_loss, outputs = self._run(
                fetches=[self.model.optimizer, self.model.loss,
                         self.model.l1_loss,
                         self.model.outputs], feed_dict=feed_dict)
        return _loss, _l1_loss, outputs

    def _predict(self, one_hot_x, multi_hot_x, y, pos_his_embs_info, pos_his_visual_info):
        one_hot_x += self.one_hot_pre_sum
        feed_dict = {self.model.labels: y,
                     self.model.one_hot_inputs: one_hot_x,}
        feed_dict[self.model.pos_main_his_embs] = pos_his_embs_info[2]
        feed_dict[self.model.pos_main_his_embs_num] = pos_his_embs_info[3]
        feed_dict[self.model.pos_slide_his_embs] = pos_his_embs_info[4]
        feed_dict[self.model.pos_slide_his_embs_num] = pos_his_embs_info[5]
        feed_dict[self.model.current_main_embs] = pos_his_embs_info[0]
        feed_dict[self.model.current_slide_embs] = pos_his_embs_info[1]

        pos_main_his_visual = pos_his_visual_info[0]
        pos_main_his_visual_mask = np.where(pos_main_his_visual >= 0, 1, 0)
        for i in range(int(self.multi_hot_field_num / 2)):
            pos_main_his_visual[:, :, i, :] += self.multi_hot_pre_sum_main[i] * pos_main_his_visual_mask[:, :, i, :]
        feed_dict[self.model.pos_main_his_visual] = pos_main_his_visual
        feed_dict[self.model.pos_main_his_visual_mask] = pos_main_his_visual_mask
        feed_dict[self.model.pos_main_his_visual_num] = pos_his_visual_info[1]
        pos_slide_his_visual = pos_his_visual_info[2]
        pos_slide_his_visual_mask = np.where(pos_slide_his_visual >= 0, 1, 0)
        for i in range(int(self.multi_hot_field_num / 2)):
            pos_slide_his_visual[:, :, i, :] += self.multi_hot_pre_sum_slide[i] * pos_slide_his_visual_mask[:, :, i, :]
        feed_dict[self.model.pos_slide_his_visual] = pos_slide_his_visual
        feed_dict[self.model.pos_slide_his_visual_mask] = pos_slide_his_visual_mask
        feed_dict[self.model.pos_slide_his_visual_num] = pos_his_visual_info[3]

        if multi_hot_x.shape[1] > 0:
            multi_hot_x_mask = np.where(multi_hot_x >= 0, 1, 0)
            for i in range(self.multi_hot_field_num):
                multi_hot_x[:, i, :] += self.multi_hot_pre_sum[i] * multi_hot_x_mask[:, i, :]
            feed_dict[self.model.multi_hot_inputs] = multi_hot_x
            feed_dict[self.model.multi_hot_feat_mask] = multi_hot_x_mask

        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = False
        return self._run(fetches=[self.model.loss, self.model.outputs],
                         feed_dict=feed_dict)

    def predict(self, gen, eval_size):
        preds = []
        labels = []
        user_ids = []
        cnt = 0
        tic = time.time()
        # for i in range(num_of_batches):
        # while cnt < num_of_batches and not finish:
        # TODO reset if this is the first iteration
        # TODO use iterator instead of generator
        for batch_data in gen:
            one_hot_x, multi_hot_x, y, x_user, date, item_ids = batch_data
            # main_item_id, slide_item_id
            if len(y.shape) > 1:
                y = np.squeeze(y)
                x_user = np.squeeze(x_user)
            if len(y.shape) != 1 or y.shape[0] == 0:
                print("Error:", y.shape)
                continue
            main_emb = np.array(list(map(self.map_item_embs, item_ids[:, 0])))
            slide_emb = np.array(list(map(self.map_item_embs, item_ids[:, 1])))
            # embed history
            main_his_embs_info = list(map(self.map_slide_item_to_main_his_embs, item_ids[:, 1], date))
            slide_his_embs_info = list(map(self.map_main_item_to_slide_his_embs, item_ids[:, 0], date))
            pos_main_his_embs = np.array([main_his_embs_info[i][0] for i in range(len(main_his_embs_info))])
            pos_main_his_embs_num = np.array([main_his_embs_info[i][1] for i in range(len(main_his_embs_info))])
            pos_slide_his_embs_num = np.array([slide_his_embs_info[i][1] for i in range(len(slide_his_embs_info))])
            pos_slide_his_embs = np.array([slide_his_embs_info[i][0] for i in range(len(slide_his_embs_info))])
            pos_his_embs_info = [main_emb, slide_emb, pos_main_his_embs, pos_main_his_embs_num, pos_slide_his_embs, pos_slide_his_embs_num]

            main_his_visual_info = list(map(self.map_slide_item_to_main_his_visual, item_ids[:, 1], date))
            slide_his_visual_info = list(map(self.map_main_item_to_slide_his_visual, item_ids[:, 0], date))
            pos_main_his_visual = np.array([main_his_visual_info[i][0] for i in range(len(main_his_visual_info))])
            pos_main_his_visual_num = np.array([main_his_visual_info[i][1] for i in range(len(main_his_visual_info))])
            pos_slide_his_visual_num = np.array([slide_his_visual_info[i][1] for i in range(len(slide_his_visual_info))])
            pos_slide_his_visual = np.array([slide_his_visual_info[i][0] for i in range(len(slide_his_visual_info))])
            pos_his_visual_info = [pos_main_his_visual, pos_main_his_visual_num, pos_slide_his_visual, pos_slide_his_visual_num]

            batch_loss, batch_pred = self._predict(one_hot_x, multi_hot_x, y, pos_his_embs_info, pos_his_visual_info)
            preds.append(batch_pred)
            labels.append(y)
            user_ids.append(x_user)
            cnt += 1
            if cnt % 10 == 0:
                print('evaluated batches:', cnt, time.time() - tic)
                tic = time.time()
        preds = np.concatenate(preds)
        preds = np.float64(preds)
        preds = np.clip(preds, 1e-8, 1 - 1e-8)
        labels = np.concatenate(labels)
        user_ids = np.concatenate(user_ids)
        group_auc = cal_group_auc(labels=labels, preds=preds,
                                  user_id_list=user_ids)
        loss = self.call_loss(y_true=labels, y_pred=preds)
        auc = self.call_auc(y_score=preds, y_true=labels)
        return labels, preds, loss, auc, group_auc

    def _batch_callback(self):
        pass

    def _epoch_callback(self, ):
        tic = time.time()
        print('running test...')
        labels, preds, loss, auc, group_auc = self.predict(self.test_gen, self.test_per_epoch)
        print('test loss = %f, test auc = %f, test group auc = %f' % (
        loss, auc, group_auc))
        toc = time.time()
        print('evaluated time:',
              str(datetime.timedelta(seconds=int(toc - tic))))
        return loss, auc

    def map_item_embs(self, x):
        try:
            x = int(float(x))
            item_actual_id = int(float(self.item_id_to_actual_item_id[x]))
            embed = self.item_embs_dict[item_actual_id]
            return embed
        except:
            return [0.0]*64

    def map_main_item_to_slide_his_embs(self, x, date):
        embed_list = []
        neg_embed_list = []
        if x in self.main_dict_set[date]:
            x_his_list = self.main_dict[date][x][0]
            for item in x_his_list:
                if item == -1 or item == 0:
                    continue
                item_actual_id = int(float(self.item_id_to_actual_item_id[item]))
                if item_actual_id in self.item_embs_set:
                    embed = self.item_embs_dict[item_actual_id]
                    embed_list.append(embed)
                    if len(embed_list) >= self.his_len:
                        break
            if self.embs_add_neg:
                x_neg_his_list = self.main_dict[date][x][1]
                for item in x_neg_his_list:
                    if item == -1 or item == 0:
                        continue
                    item_actual_id = int(float(self.item_id_to_actual_item_id[item]))
                    if item_actual_id in self.item_embs_set:
                        embed = self.item_embs_dict[item_actual_id]
                        neg_embed_list.append(embed)
                        if len(neg_embed_list) >= self.his_len:
                            break
        click_length = len(embed_list)
        embed_list += [[0.0]*64]*(self.his_len-click_length)
        neg_click_length = len(neg_embed_list)
        if self.embs_add_neg:
            neg_embed_list += [[0.0]*64]*(self.his_len-neg_click_length)
        return embed_list, click_length, neg_embed_list, neg_click_length

    def map_slide_item_to_main_his_embs(self, x, date):
        embed_list = []
        neg_embed_list = []
        if x in self.slide_dict_set[date]:
            x_his_list = self.slide_dict[date][x][0]
            for item in x_his_list:
                if item == -1 or item == 0:
                    continue
                item_actual_id = int(float(self.item_id_to_actual_item_id[item]))
                if item_actual_id in self.item_embs_set:
                    embed = self.item_embs_dict[item_actual_id]
                    embed_list.append(embed)
                    if len(embed_list) >= self.his_len:
                        break
            if self.embs_add_neg:
                x_neg_his_list = self.slide_dict[date][x][1]
                for item in x_neg_his_list:
                    if item == -1 or item == 0:
                        continue
                    item_actual_id = int(float(self.item_id_to_actual_item_id[item]))
                    if item_actual_id in self.item_embs_set:
                        embed = self.item_embs_dict[item_actual_id]
                        neg_embed_list.append(embed)
                        if len(neg_embed_list) >= self.his_len:
                            break
        click_length = len(embed_list)
        embed_list += [[0.0]*64]*(self.his_len-click_length)
        neg_click_length = len(neg_embed_list)
        if self.embs_add_neg:
            neg_embed_list += [[0.0]*64]*(self.his_len-neg_click_length)
        return embed_list, click_length, neg_embed_list, neg_click_length

    def map_main_item_to_slide_his_visual(self, x, date):
        visual_list = []
        neg_visual_list = []
        if x in self.main_dict_set[date]:
            x_his_list = self.main_dict[date][x][0]
            for item in x_his_list:
                if item == -1 or item == 0:
                    continue
                if item in self.visual_set:
                    info = self.visual_dict[item]
                    visual_list.append(info)
                    if len(visual_list) >= self.his_len:
                        break
            if self.visual_add_neg:
                x_neg_his_list = self.main_dict[date][x][1]
                for item in x_neg_his_list:
                    if item == -1 or item == 0:
                        continue
                    if item in self.visual_set:
                        info = self.visual_dict[item]
                        neg_visual_list.append(info)
                        if len(neg_visual_list) >= self.his_len:
                            break
        click_length = len(visual_list)
        visual_list += [[[-1]*30]*3]*(self.his_len-click_length)
        neg_click_length = len(neg_visual_list)
        if self.visual_add_neg:
            neg_visual_list += [[[-1]*30]*3]*(self.his_len-neg_click_length)
        return visual_list, click_length, neg_visual_list, neg_click_length

    def map_slide_item_to_main_his_visual(self, x, date):
        visual_list = []
        neg_visual_list = []
        if x in self.slide_dict_set[date]:
            x_his_list = self.slide_dict[date][x][0]
            for item in x_his_list:
                if item == -1 or item == 0:
                    continue
                if item in self.visual_set:
                    info = self.visual_dict[item]
                    visual_list.append(info)
                    if len(visual_list) >= self.his_len:
                        break
            if self.visual_add_neg:
                x_neg_his_list = self.slide_dict[date][x][1]
                for item in x_neg_his_list:
                    if item == -1 or item == 0:
                        continue
                    if item in self.visual_set:
                        info = self.visual_dict[item]
                        neg_visual_list.append(info)
                        if len(neg_visual_list) >= self.his_len:
                            break
        click_length = len(visual_list)
        visual_list += [[[-1]*30]*3]*(self.his_len-click_length)
        neg_click_length = len(neg_visual_list)
        if self.visual_add_neg:
            neg_visual_list += [[[-1]*30]*3]*(self.his_len-neg_click_length)
        return visual_list, click_length, neg_visual_list, neg_click_length

    def fit(self):
        num_of_batches = int(
            np.ceil(self.train_per_epoch / self.batch_size)) + 1
        total_batches = self.n_epoch * num_of_batches
        print('total batches: %d\tbatch per epoch: %d' % (
        total_batches, num_of_batches))
        start_time = time.time()
        epoch = 1
        finished_batches = 0
        avg_loss = 0
        avg_l1 = 0
        label_list = []
        pred_list = []
        loss_list = []
        auc_list = []

        test_every_epoch = self.test_every_epoch

        while epoch <= self.n_epoch:
            print('======new iteration======')
            epoch_batches = 0
            for batch_data in self.train_gen:
                one_hot_x, multi_hot_x, y, x_user, date, item_ids = batch_data
                # main_item_id, slide_item_id
                if len(y.shape) > 1:
                    y = np.squeeze(y)
                    x_user = np.squeeze(x_user)
                if len(y.shape) != 1 or y.shape[0] == 0:
                    print("Error:", y.shape)
                    continue

                main_emb = np.array(list(map(self.map_item_embs, item_ids[:, 0])))
                slide_emb = np.array(list(map(self.map_item_embs, item_ids[:, 1])))
                # embed history
                main_his_embs_info = list(map(self.map_slide_item_to_main_his_embs, item_ids[:, 1], date))
                slide_his_embs_info = list(map(self.map_main_item_to_slide_his_embs, item_ids[:, 0], date))
                pos_main_his_embs = np.array([main_his_embs_info[i][0] for i in range(len(main_his_embs_info))])
                pos_main_his_embs_num = np.array([main_his_embs_info[i][1] for i in range(len(main_his_embs_info))])
                pos_slide_his_embs_num = np.array([slide_his_embs_info[i][1] for i in range(len(slide_his_embs_info))])
                pos_slide_his_embs = np.array([slide_his_embs_info[i][0] for i in range(len(slide_his_embs_info))])
                pos_his_embs_info = [main_emb, slide_emb, pos_main_his_embs, pos_main_his_embs_num, pos_slide_his_embs, pos_slide_his_embs_num]
                # visual history
                main_his_visual_info = list(map(self.map_slide_item_to_main_his_visual, item_ids[:, 1], date))
                slide_his_visual_info = list(map(self.map_main_item_to_slide_his_visual, item_ids[:, 0], date))
                pos_main_his_visual = np.array([main_his_visual_info[i][0] for i in range(len(main_his_visual_info))])
                pos_main_his_visual_num = np.array([main_his_visual_info[i][1] for i in range(len(main_his_visual_info))])
                pos_slide_his_visual_num = np.array([slide_his_visual_info[i][1] for i in range(len(slide_his_visual_info))])
                pos_slide_his_visual = np.array([slide_his_visual_info[i][0] for i in range(len(slide_his_visual_info))])
                pos_his_visual_info = [pos_main_his_visual, pos_main_his_visual_num, pos_slide_his_visual, pos_slide_his_visual_num]

                batch_loss, batch_l1, batch_pred = self._train(one_hot_x, multi_hot_x, y, pos_his_embs_info, pos_his_visual_info)
                label_list.append(y)
                pred_list.append(batch_pred)
                avg_loss += batch_loss
                avg_l1 += batch_l1
                finished_batches += 1
                epoch_batches += 1

                epoch_batch_num = 100
                if epoch_batches % epoch_batch_num == 0:
                    avg_loss /= epoch_batch_num
                    avg_l1 /= epoch_batch_num
                    label_list = np.concatenate(label_list)
                    pred_list = np.concatenate(pred_list)
                    moving_auc = self.call_auc(y_true=label_list,
                                               y_score=pred_list)
                    elapsed = int(time.time() - start_time)
                    eta = int((
                                          total_batches - finished_batches) / finished_batches * elapsed)
                    print("elapsed : %s, ETA : %s" % (str(datetime.timedelta(seconds=elapsed)),
                                                      str(datetime.timedelta(seconds=eta))))
                    print('epoch %d / %d, batch %d / %d, global_step = %d, learning_rate = %e, loss = %f, l1 = %f, '
                          'auc = %f' % (epoch, self.n_epoch, epoch_batches, num_of_batches,
                                        self.global_step.eval(self.session), self._learning_rate,
                                        avg_loss, avg_l1, moving_auc))
                    label_list = []
                    pred_list = []
                    avg_loss = 0
                    avg_l1 = 0

                if epoch_batches % num_of_batches == 0:
                    if epoch % test_every_epoch == 0:
                        l, a = self._epoch_callback()
                        loss_list.append(l)
                        auc_list.append(a)
                    self._learning_rate *= self.decay_rate
                    epoch += 1
                    epoch_batches = 0
                    if epoch > self.n_epoch:
                        return

            if epoch_batches % num_of_batches != 0:
                if epoch % test_every_epoch == 0:
                    l, a = self._epoch_callback()
                    loss_list.append(l)
                    auc_list.append(a)
                self._learning_rate *= self.decay_rate
                epoch += 1
                epoch_batches = 0
                if epoch > self.n_epoch:
                    return
