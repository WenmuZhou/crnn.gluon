# -*- coding: utf-8 -*-
# @Time    : 18-11-28 上午11:20
# @Author  : zhoujun
import keys

trainfile = 'E:/zj/dataset/train.csv'
testfile = 'E:/zj/dataset/train.csv'
output_dir = 'output/test'

gpu_id = 0
workers = 0
start_epoch = 0
end_epoch = 100

train_batch_size = 4
eval_batch_size = 4
# img shape
img_h = 32
img_w = 320
img_channel = 3
nh = 512

lr = 0.001
end_lr = 1e-7
lr_decay = 0.1
lr_decay_step = 15
alphabet = keys.no_alphabet
display_interval = 10
restart_training = True
checkpoint = ''

# random seed
seed = 2
