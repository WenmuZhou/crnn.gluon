# -*- coding: utf-8 -*-
# @Time    : 18-11-28 上午11:20
# @Author  : zhoujun
import keys

trainfile = '/data/zhy/crnn/Chinese_character/train2.txt'
testfile = '/data/zhy/crnn/Chinese_character/test2.txt'
output_dir = 'output/test'

gpu_id = 3
workers = 6
start_epoch = 0
end_epoch = 100

train_batch_size = 128
eval_batch_size = 64
# img shape
img_h = 32
img_w = 320
img_channel = 3
nh = 512

lr = 0.001
end_lr = 1e-7
lr_decay = 0.1
lr_decay_step = 15
alphabet = keys.txt_alphabet
display_interval = 100
restart_training = True
checkpoint = 'output/resnet_pytorch/13_0.9237533417290824.pth'

# random seed
seed = 2