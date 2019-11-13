# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 15:31
# @Author  : zhoujun

""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import lmdb
import cv2
from tqdm import tqdm
import numpy as np

from data_loader import get_datalist
from utils import punctuation_mend


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(data_list, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list, desc='make dataset, save to {}'.format(outputPath)):
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    data_list = [["/media/zj/资料/zj/dataset/train_linux.csv"]]
    save_path = './lmdb/train'
    os.makedirs(save_path, exist_ok=True)
    train_data_list, val_data_list = get_datalist(data_list, val_data_path=data_list[0])
    train_data_list = train_data_list[0]
    alphabet = [x[1] for x in train_data_list]
    alphabet.extend([x[1] for x in val_data_list])
    alphabet = [punctuation_mend(x) for x in alphabet]
    alphabet = ''.join(sorted(set((''.join(alphabet)))))
    alphabet.replace(' ', '')
    np.save(os.path.join(save_path, 'alphabet.npy'), alphabet)
    createDataset(train_data_list, save_path)
    createDataset(val_data_list, save_path.replace('train', 'validation'))
