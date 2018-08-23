# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:21
# @Author  : zhoujun

import math
import os
import time
import random
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from mxnet.gluon.data import ArrayDataset, DataLoader
import string
import tarfile
import urllib
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import cv2
import numpy as np
import _pickle as cPickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from leven import levenshtein
import glob
import matplotlib.gridspec as gridspec
from unit import *
from data_iter import Gluon_OCRIter

DESIRED_SIZE = (32, 280)
IMAGES_DATA_FILE = '/home/han/data/ocr/id_date/images_data.pkl'
DATA_FOLDER = '/home/han/data/ocr/id_date/'
Train_file = '/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/data/ocr/id_date/id_date_train_.txt'
Test_file = '/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/data/ocr/id_date/id_date_test_.txt'
DATA_ROOT = '/home/han/data/ocr/id_date/id_riqi/'


# This decodes the predictions and the labels back to words
def decode(prediction):
    results = []
    for word in prediction:
        result = []
        for i, index in enumerate(word):
            if i < len(word) - 1 and word[i] == word[i + 1] and word[-1] != -1:  # Hack to decode label as well
                continue
            if index == len(ALPHABET) or index == -1:
                continue
            else:
                result.append(ALPHABET[int(index)])
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


def plot_predictions(images, predictions, labels):
    gs = gridspec.GridSpec(6, 2)
    fig = plt.figure(figsize=(15, 10))
    gs.update(hspace=0.1, wspace=0.1)
    for gg, prediction, label, image in zip(gs, predictions, labels, images):
        gg2 = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=gg)
        ax = fig.add_subplot(gg2[:, :])
        ax.imshow(image.asnumpy().squeeze(), cmap='Greys_r')
        ax.tick_params(axis='both',
                       which='both',
                       bottom='off',
                       top='off',
                       left='off',
                       right='off',
                       labelleft='off',
                       labelbottom='off')
        ax.axes.set_title("{} | {}".format(label, prediction))
    plt.show()


if __name__ == '__main__':
    SEQ_LEN = 32
    ALPHABET = string.digits + '-' + '.'
    ALPHABET_INDEX = {ALPHABET[i]: i for i in range(len(ALPHABET))}

    ALPHABET_SIZE = len(ALPHABET) + 1
    BATCH_SIZE = 64

    dataset_test = Gluon_OCRIter(
        DATA_ROOT, Test_file, DESIRED_SIZE, SEQ_LEN, )

    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, last_batch='discard', shuffle=True)

    NUM_HIDDEN = 200
    NUM_CLASSES = 13
    NUM_LSTM_LAYER = 1
    p_dropout = 0.5

    ctx = mx.gpu(0)
    net2 = get_net()
    net2.load_params('/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/model/ocr/crnn/6-crnn.params', ctx)
    # print('Net 2:', evaluate_accuracy(net2, dataloader_test))
    for i, (data, label) in enumerate(dataloader_test):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net2(data)
        break
    predictions = decode(output.softmax().topk(axis=2).asnumpy())
    labels = decode(label.asnumpy())
    plot_predictions(data, predictions, labels)