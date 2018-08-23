# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun

import time
import random
import os
import matplotlib.pyplot as plt
import argparse
import string, codecs
import logging
import mxnet as mx
import numpy as np
from skimage import transform as skimage_tf
from skimage import exposure
from leven import levenshtein
from mxnet import nd, autograd, gluon
from mxboard import SummaryWriter
from mxnet.gluon.model_zoo.vision import resnet34_v1

np.seterr(all='raise')

import multiprocessing

mx.random.seed(1)

from utils.iam_dataset import IAMDataset
from utils.draw_text_on_image import draw_text_on_image
from data_iter import Gluon_OCRIter

DESIRED_SIZE = (32, 280)
# Train_file='./data/train.txt'
# Test_file='./data/test.txt'
# DATA_ROOT='./data/images/'
# All_file ='./data/all.txt'
DATA_ROOT = '/home/han/data/ocr/id_date/id_riqi/'
Train_file = './id_date_train_.txt'
Test_file = './id_date_test_.txt'
All_file = './all.txt'

LabelFile = './data/char_std_5990.txt'


def transform(image, label):
    image = skimage_tf.resize(image, (30, 400), mode='constant')
    image = np.expand_dims(image, axis=0).astype(np.float32)
    if image[0, 0, 0] > 1:
        image = image / 255.

    label_encoded = np.zeros(max_seq_len, dtype=np.float32) - 1
    # i = 0
    # for letter in label[0]:
    #     label_encoded[i] = alphabet_dict[letter]
    #     i += 1

    i = 0
    for word in label:
        # if i >= max_seq_len:
        #     break
        for letter in word:
            label_encoded[i] = letter  # alphabet_dict[letter]
            i += 1
    return image, label_encoded


def augment_transform(image, label):
    ty = random.uniform(-random_y_translation, random_y_translation)
    tx = random.uniform(-random_x_translation, random_x_translation)

    sx = random.uniform(1. - random_y_scaling, 1. + random_y_scaling)
    sy = random.uniform(1. - random_x_scaling, 1. + random_x_scaling)

    s = random.uniform(-random_shearing, random_shearing)

    st = skimage_tf.AffineTransform(scale=(sx, sy),
                                    shear=s,
                                    translation=(tx * image.shape[1], ty * image.shape[0]))
    augmented_image = skimage_tf.warp(image, st, cval=1.0)
    return transform(augmented_image, label)


def decode(prediction):
    results = []
    for word in prediction:
        result = []
        for i, index in enumerate(word):
            if i < len(word) - 1 and word[i] == word[i + 1] and word[-1] != -1:  # Hack to decode label as well
                continue
            if index == len(alphabet_dict) or index == -1:
                continue
            else:
                result.append(alphabet_encoding[int(index)])
        results.append(result)
    words = [''.join(word) for word in results]
    return words


def metric_levenshtein(predictions, labels):
    predictions = predictions.softmax().topk(axis=2).asnumpy()
    zipped = zip(decode(labels.asnumpy()), decode(predictions))
    metric = sum([(len(label) - levenshtein(label, pred)) / len(label) for label, pred in zipped])
    return metric / len(labels)


def evaluate_accuracy(net, dataloader):
    metric = 0
    for i, (data, label) in enumerate(dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric += metric_levenshtein(output, label)
    return metric / (i + 1)


def run_epoch(logger, e, network, dataloader, trainer, log_dir, print_name, update_network, save_network):
    total_loss = nd.zeros(1, ctx)
    tick = time.time()
    loss = nd.zeros(1, ctx)
    for i, (x, y) in enumerate(dataloader):
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)

        with autograd.record():
            output = network(x)
            loss_ctc = ctc_loss(output, y)

        if update_network:
            loss_ctc.backward()
            trainer.step(x.shape[0])
        # if i == 0 and e % send_image_every_n == 0 and e > 0:
        #     predictions = output.softmax().topk(axis=2).asnumpy()
        #     decoded_text = decode(predictions)
        #     output_image = draw_text_on_image(x.asnumpy(), decoded_text)
        #     print("{} first decoded text = {}".format(print_name, decoded_text[0]))
        #     with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        #         sw.add_image('bb_{}_image'.format(print_name), output_image, global_step=e)

        total_loss += loss_ctc.mean()
        loss += loss_ctc.mean()
        if i % print_n == 0 and i > 0:
            logger.info('Batches {0}: CTC Loss: {1:.2f}, time:{2:.2f} s'.format(
                i, float(loss.asscalar() / print_n), time.time() - tick))
            loss = nd.zeros(1, ctx)
            tick = time.time()
            nd.waitall()
    epoch_loss = float(total_loss.asscalar()) / len(dataloader)

    with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)

    if save_network and e % save_every_n == 0 and e > 0:
        network.save_parameters("{}/{}".format(checkpoint_dir, checkpoint_name))

    return epoch_loss


if __name__ == '__main__':
    # for line in open(LabelFile,'rb'):
    #     str_ += line.strip().decode('gb18030')
    a = codecs.open(LabelFile, 'r', encoding='utf-8')
    str_ = ''
    for line in a.readlines():
        str_ += line.strip()
    # SEQ_LEN = 32
    # ALPHABET_SIZE =len(ALPHABET)+1
    alphabet_encoding = string.digits + '-' + '.' + ' '
    # alphabet_encoding =str_
    alphabet_dict = {alphabet_encoding[i]: i for i in range(len(alphabet_encoding))}

    max_seq_len = 32
    print_every_n = 1
    print_n = 50
    save_every_n = 2
    send_image_every_n = 10

    epochs = 120
    learning_rate = 0.0001
    batch_size = 128

    num_downsamples = 2
    resnet_layer_id = 4
    lstm_hidden_states = 200
    lstm_layers = 1
    ctx = mx.gpu(1)

    random_y_translation, random_x_translation = 0.03, 0.03
    random_y_scaling, random_x_scaling = 0.1, 0.1
    random_shearing = 0.5

    log_dir = "./logs"
    checkpoint_dir = "model_checkpoint"
    checkpoint_name = "handwriting.params"
    # train_ds = IAMDataset("line", output_data="text", train=True,\
    #     data_root=DATA_ROOT , data_list=All_file,data_list_tra=Train_file,data_list_test=Test_file,num_label=30)
    # print("Number of training samples: {}".format(len(train_ds)))

    # test_ds = IAMDataset("line", output_data="text", train=False,\
    #     data_root=DATA_ROOT , data_list=All_file,data_list_tra=Train_file,data_list_test=Test_file,num_label=30 )
    # print("Number of testing samples: {}".format(len(test_ds)))

    # train_data = gluon.data.DataLoader(train_ds.transform(augment_transform), batch_size, shuffle=True, last_batch="discard")
    # test_data = gluon.data.DataLoader(test_ds.transform(transform), batch_size, shuffle=False, last_batch="discard")#, num_workers=multiprocessing.cpu_count()-2)

    dataset_train = Gluon_OCRIter(
        DATA_ROOT, Train_file, DESIRED_SIZE, max_seq_len, )
    dataset_test = Gluon_OCRIter(
        DATA_ROOT, Test_file, DESIRED_SIZE, max_seq_len, )
    train_data = gluon.data.DataLoader(dataset_train, batch_size=batch_size, last_batch='discard', shuffle=True)
    train_data = gluon.data.DataLoader(dataset_test, batch_size=batch_size, last_batch='discard', shuffle=True)

    net = Network(num_downsamples=num_downsamples, resnet_layer_id=resnet_layer_id,
                  lstm_hidden_states=lstm_hidden_states, lstm_layers=lstm_layers)
    net.hybridize()

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

    ctc_loss = gluon.loss.CTCLoss(weight=0.2)
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = 'train.log'
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    for e in range(epochs):
        train_loss = run_epoch(logger, e, net, train_data, trainer, log_dir, print_name="train",
                               update_network=True, save_network=True)
        test_loss = run_epoch(logger, e, net, test_data, trainer, log_dir, print_name="test",
                              update_network=False, save_network=False)
        if e % print_every_n == 0 and e > 0:
            # print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))
            logger.info("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))
        validation_accuracy = evaluate_accuracy(net, test_data)
        logger.info("Epoch {0}, Val_acc {1:.2f}".format(e, validation_accuracy))