import os
from tf_models import ComboFashion
from tf_trainer import Trainer
import tensorflow as tf
import __init__
import sys
sys.path.append(__init__.config['data_path'])
backend = 'tf'
from Detail_data import Detail_data


DATA_CONFIG = {
    'TRAIN': {
        'part': 'train',
        'shuffle': True,
        'batch_size': 1024,
        'portion': 1.0
    },
    'TEST': {
        'part': 'test',
        'shuffle': False,
        'batch_size': 1024,
    },
}


def run_one_model(dataset, model=None, learning_rate=1e-3, decay_rate=1.0,
                  epsilon=1e-8, ep=5, his_len=50):
    n_ep = ep * 2
    train_param = {
        'opt': 'adam',
        'loss': 'weight',
        'pos_weight': 1.0,
        'n_epoch': n_ep,
        'train_per_epoch': dataset.train_size / ep,  # split training data
        'test_per_epoch': dataset.test_size,
        'early_stop_epoch': int(0.5 * ep),
        'test_every_epoch': int(ep / 5),
        'batch_size': DATA_CONFIG['TRAIN']['batch_size'],
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'epsilon': epsilon,
        'one_hot_pre_sum': dataset.one_hot_pre_sum,
        'multi_hot_pre_sum': dataset.multi_hot_pre_sum,
        'his_len': his_len,
    }
    train_gen = dataset.batch_generator(DATA_CONFIG['TRAIN'])
    test_gen = dataset.batch_generator(DATA_CONFIG['TEST'])
    trainer = Trainer(model=model, train_gen=train_gen, test_gen=test_gen,
                      **train_param)
    trainer.fit()
    trainer.session.close()


def train():
    embed_size = 4
    multi_hot_embed_size = 4
    learning_rate = 1e-3
    dc = 0.7
    his_len = 20
    batch_norm = True
    layer_norm = False
    l1_w = 0.0
    l1_v = 0.0
    layer_l1 = 0.0
    layer_sizes = [256, 128, 64, 1]
    layer_acts = ['relu', 'relu', 'relu', None]
    layer_keeps = [1.0, 1.0, 1.0, 1.0]
    dataset = Detail_data(train_part="7day")
    split_epoch = 5
    model = ComboFashion(init="xavier", input_dim=dataset.num_features, multi_hot_input_dim=dataset.multi_num_features,
                         embed_size=embed_size, multi_hot_embed_size=multi_hot_embed_size,
                         layer_sizes=layer_sizes, layer_acts=layer_acts, layer_keeps=layer_keeps,
                         l1_w=l1_w, l1_v=l1_v, layer_l1=layer_l1, multi_hot_length=dataset.len_field_multi_hot,
                         one_hot_field_num=dataset.num_field_one_hot, multi_hot_field_num=dataset.num_field_multi_hot,
                         batch_norm=batch_norm, layer_norm=layer_norm, his_len=his_len)
    run_one_model(dataset, model=model, learning_rate=learning_rate, epsilon=1e-8,
                  decay_rate=dc, ep=split_epoch, his_len=his_len)
    tf.reset_default_graph()
    del dataset


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
