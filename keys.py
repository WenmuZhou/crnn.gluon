# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 10:20
# @Author  : zhoujun

# NOTE: -1 is reserved for 'blank' required by mxnet ctc
BLANK_SYMBOL = '_'
alphabet = u'0123456789.-\/:;D,wsCc()zo+v' + BLANK_SYMBOL

import os
if __name__ == '__main__':
    txt_path = r'/data1/lsp/lsp/number_crnn/crnn/data/test.txt'
    lines = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if not os.path.exists(line[0]):
                print(line[0])