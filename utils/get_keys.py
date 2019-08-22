# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 10:20
# @Author  : zhoujun
import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(os.path.abspath(__name__)).parent))
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from utils import load_json


def get_key(label_file_list, show_max_img=False):
    data_list = []
    label_list = []
    max_len = 0
    max_h = 0
    max_w = 0
    for label_path in label_file_list:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc=label_path):
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    data_list.append(line[0])
                    label_list.append(line[1])
                    max_len = max(max_len, len(line[1]))
                    if show_max_img:
                        img = cv2.imread(line[0])
                        h, w = img.shape[:2]
                        max_h = max(max_h, h)
                        max_w = max(max_w, w)
    if show_max_img:
        print('max len of label is {}, max img_h is {}, max img_w is {}'.format(max_len, max_h, max_w))
    a = ''.join(sorted(set((''.join(label_list)))))
    return a


if __name__ == '__main__':
    # 根据label文本生产key
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_file', nargs='+', help='label file', default=[r"E:\\zj\\dataset\\train.csv"])
    args = parser.parse_args()

    if os.path.exists('config.json'):
        data = load_json('config.json')
        label_file = []
        for train_file in data['data_loader']['args']['dataset']['train_data_path']:
            if isinstance(train_file, list):
                label_file.extend(train_file)
            else:
                label_file.append(train_file)
        label_file.extend(data['data_loader']['args']['dataset']['val_data_path'])
    else:
        label_file = args.label_file
    alphabet = get_key(label_file).replace(' ', '') + '嫑'
    np.save('alphabet.npy', alphabet)
