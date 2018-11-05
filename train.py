# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun
import os
import time
import shutil
import argparse
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter
import keys
from dataset import ImageDataset
from crnn import CRNN
from predict import decode


def accuracy(predictions, labels, alphabet):
    predictions = predictions.softmax().topk(axis=2).asnumpy()
    zipped = zip(decode(predictions, alphabet), decode(labels.asnumpy(), alphabet))
    n_correct = 0
    for pred, target in zipped:
        if pred == target:
            n_correct += 1
    return n_correct


def evaluate_accuracy(net, dataloader, ctx, alphabet):
    metric = 0
    for i, (data, label) in enumerate(dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric += accuracy(output, label, alphabet)
    return metric


def train(opt):
    if opt.restart_training:
        shutil.rmtree(opt.output_dir, ignore_errors=True)
    if opt.output_dir is None:
        opt.output_dir = 'output'
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    ctx = mx.gpu(opt.gpu)
    # 设置随机种子
    mx.random.seed(2)
    mx.random.seed(2, ctx=ctx)

    train_transfroms = transforms.Compose([
        transforms.RandomBrightness(0.5),
        transforms.ToTensor()
    ])
    train_dataset = ImageDataset(opt.trainfile, (opt.imgH, opt.imgW), 3, 80, opt.alphabet, phase='train')
    train_data_loader = DataLoader(train_dataset.transform_first(train_transfroms), opt.batchSize, shuffle=True,
                                   last_batch='keep', num_workers=opt.workers)
    test_dataset = ImageDataset(opt.testfile, (opt.imgH, opt.imgW), 3, 80, opt.alphabet, phase='test')
    test_data_loader = DataLoader(test_dataset.transform_first(transforms.ToTensor()), opt.batchSize, shuffle=True,
                                  last_batch='keep', num_workers=opt.workers)
    net = CRNN(len(opt.alphabet), hidden_size=opt.nh)
    net.hybridize()
    if opt.model:
        print('load pretrained net from {}'.format(opt.model))
        net.load_parameters(opt.model, ctx=ctx)
    else:
        net.initialize(ctx=ctx)

    criterion = gluon.loss.CTCLoss()

    all_step = train_dataset.__len__() // opt.batchSize
    schedule = mx.lr_scheduler.FactorScheduler(step=15 * all_step, factor=0.1, stop_factor_lr=opt.end_lr)
    # schedule = mx.lr_scheduler.MultiFactorScheduler(step=[15 * all_step, 30 * all_step, 60 * all_step,80 * all_step],
    #                                                 factor=0.1)
    adam_optimizer = mx.optimizer.Adam(learning_rate=opt.lr, lr_scheduler=schedule)
    trainer = gluon.Trainer(net.collect_params(), optimizer=adam_optimizer)

    sw = SummaryWriter(logdir=opt.output_dir, flush_secs=5)
    for epoch in range(opt.start_epochs, opt.epochs):
        loss = .0
        train_acc = .0
        tick = time.time()
        cur_step = 0
        for i, (data, label) in enumerate(train_data_loader):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                output = net(data)
                loss_ctc = criterion(output, label)
            loss_ctc.backward()
            trainer.step(data.shape[0])

            loss_c = loss_ctc.mean()
            cur_step = epoch * all_step + i
            sw.add_scalar(tag='ctc_loss', value=loss_c.asscalar(), global_step=cur_step // 2)
            sw.add_scalar(tag='lr', value=trainer.learning_rate, global_step=cur_step // 2)
            loss += loss_c
            acc = accuracy(output, label, opt.alphabet)
            train_acc += acc
            if (i + 1) % opt.displayInterval == 0:
                acc /= len(label)
                sw.add_scalar(tag='train_acc', value=acc, global_step=cur_step)
                print('[{}/{}], [{}/{}], ctc loss: {:.4f},acc: {:.4f}, lr:{}, time:{:.4f} s'.format(
                    epoch, opt.epochs, i, all_step, loss.asscalar() / opt.displayInterval, acc,
                    trainer.learning_rate, time.time() - tick))
                loss = .0
                tick = time.time()
                nd.waitall()
        if epoch == 0:
            sw.add_graph(net)
        print('start val ....')
        train_acc /= train_dataset.__len__()
        validation_accuracy = evaluate_accuracy(net, test_data_loader, ctx, opt.alphabet) / test_dataset.__len__()
        sw.add_scalar(tag='val_acc', value=validation_accuracy, global_step=cur_step)
        print("Epoch {},train_acc {:.4f}, val_acc {:.4f}".format(epoch, train_acc, validation_accuracy))
        net.save_parameters("{}/{}_{:.4f}_{:.4f}.params".format(opt.output_dir, epoch, train_acc, validation_accuracy))
    sw.close()


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', default='/data/zhy/crnn/Chinese_character/train2.txt',
                        # /data1/zj/data/no/train.txt
                        help='path to train dataset file')
    parser.add_argument('--testfile', default='/data/zhy/crnn/Chinese_character/test2.txt',
                        # /data1/zj/data/no/test.txt
                        help='path to test dataset file')

    parser.add_argument('--gpu', type=int, default=3, help='the gpu id')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--start_epochs', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=320, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=512, help='size of the lstm hidden state')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--end_lr', type=float, default=1e-7, help='the end learning rate')
    parser.add_argument('--alphabet', type=str, default=keys.txt_alphabet)
    parser.add_argument('--output_dir', default='output/crnn_lstm_txt_resnet_2048_dropout_lstm_512_data_augment2',
                        help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')

    parser.add_argument('--model', default='', help="path to crnn (to continue training)")
    parser.add_argument('--restart_training', type=bool, default=True,
                        help="Restart from step 1 and remove summaries and checkpoints.")

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = init_args()
    print('train with gpu %s and mxnet %s' % (opt.gpu, mx.__version__))
    print(opt)
    train(opt)
