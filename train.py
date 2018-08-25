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
from dataset import Gluon_OCRDataset
from crnn import CRNN
from predict import decode


def accuracy(predictions, labels, alphabet):
    predictions = predictions.softmax().topk(axis=2).asnumpy()
    zipped = zip(decode(predictions, alphabet), decode(labels.asnumpy(), alphabet))
    n_correct = 0
    for pred, target in zipped:
        if pred == target:
            n_correct += 1
    return n_correct / len(labels)


def evaluate_accuracy(net, dataloader, ctx, alphabet):
    metric = 0
    for i, (data, label) in enumerate(dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric += accuracy(output, label, alphabet)
    return metric / dataloader.__len__()


def train(opt):
    if opt.restart_training:
        shutil.rmtree(opt.output_dir, ignore_errors=True)
    if opt.output_dir is None:
        opt.output_dir = 'output'
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    ctx = mx.gpu(opt.gpu)
    train_dataset = Gluon_OCRDataset(opt.trainfile, (opt.imgH, opt.imgW), 3, 81, opt.alphabet)
    train_data_loader = DataLoader(train_dataset.transform_first(transforms.ToTensor()), opt.batchSize, shuffle=True,
                                   last_batch='keep', num_workers=opt.workers)
    test_dataset = Gluon_OCRDataset(opt.trainfile, (opt.imgH, opt.imgW), 3, 81, opt.alphabet)
    test_data_loader = DataLoader(test_dataset.transform_first(transforms.ToTensor()), opt.batchSize, shuffle=True,
                                  last_batch='keep', num_workers=opt.workers)
    net = CRNN(len(opt.alphabet), hidden_size=opt.nh)
    net.hybridize()
    if opt.model:
        print('load pretrained net from {}'.format(opt.model))
        net.load_parameters(opt.model, ctx=ctx)
    else:
        net.initialize(ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr, })
    sw = SummaryWriter(logdir=opt.output_dir, flush_secs=5)
    criterion = gluon.loss.CTCLoss()
    all_step = train_dataset.__len__() // opt.batchSize
    for epoch in range(opt.start_epochs, opt.epochs):
        loss = .0
        tick = time.time()
        acc = .0
        cur_step = 0
        if (epoch + 1) % 10 == 0 and trainer.learning_rate > opt.end_lr and epoch > opt.start_epochs:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
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
            sw.add_scalar(tag='ctc_loss', value=loss_c.asscalar(), global_step=cur_step)
            sw.add_scalar(tag='lr', value=trainer.learning_rate, global_step=cur_step)
            loss += loss_c

            if (i + 1) % opt.displayInterval == 0:
                acc += accuracy(output, label, keys.alphabet)
                sw.add_scalar(tag='train_acc', value=acc, global_step=cur_step)
                print('[{}/{}], [{}/{}], CTC Loss: {:.4f},acc: {}, lr:{}, time:{:.4f} s'.format(epoch + 1, opt.epochs,
                                                                                                i + 1, all_step,
                                                                                                loss.asscalar() / opt.displayInterval,
                                                                                                acc,
                                                                                                trainer.learning_rate,
                                                                                                time.time() - tick))
                loss = .0
                acc = .0
                tick = time.time()
                nd.waitall()
        if epoch == 0:
            sw.add_graph(net)
        print('start val ....')
        validation_accuracy = evaluate_accuracy(net, test_data_loader, ctx, opt.alphabet)
        sw.add_scalar(tag='val_acc', value=validation_accuracy, global_step=cur_step)
        print("Epoch {0}, Val_acc {1:.2f}".format(epoch, validation_accuracy))
        net.save_parameters("{}/{}_{}.params".format(opt.output_dir, epoch + 1, validation_accuracy))
    sw.close()


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', default='/data1/zj/data/crnn/train.txt',
                        help='path to train dataset file')
    parser.add_argument('--testfile', default='/data1/zj/data/crnn/test.txt',
                        help='path to test dataset file')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--start_epochs', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=320, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--end_lr', type=float, default=1e-6, help='the end learning rate')
    parser.add_argument('--model', default='',
                        help="path to crnn (to continue training)")
    parser.add_argument('--alphabet', type=str, default=keys.alphabet)
    parser.add_argument('--output_dir', default='output/gru_crnn', help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
    parser.add_argument('--restart_training', type=bool, default=True,
                        help="Restart from step 1 and remove summaries and checkpoints.")

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = init_args()
    print('train with gpu %s and mxnet %s' % (opt.gpu, mx.__version__))
    print(opt)
    train(opt)
