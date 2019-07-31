# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun
import os
import time
import shutil
from tqdm import tqdm
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter
import config
from dataset import ImageDataset
from crnn import CRNN
from predict import decode1,decode,try_gpu


def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger


def accuracy(predictions, labels, alphabet):
    predictions = predictions.softmax().asnumpy()
    zipped = zip(decode1(predictions, alphabet), decode1(labels.asnumpy(), alphabet))
    n_correct = 0
    for (pred, pred_conf), (target, _) in zipped:
        if pred == target:
            n_correct += 1
    return n_correct


def evaluate_accuracy(net, dataloader, ctx, alphabet):
    metric = 0
    for data, label in tqdm(dataloader,desc='test model'):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric += accuracy(output, label, alphabet)
    return metric


def train():
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if config.output_dir is None:
        config.output_dir = 'output'
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))
    ctx =try_gpu(config.gpu_id)
    logger.info('train with %s and mxnet %s' % (ctx, mx.__version__))

    # 设置随机种子
    mx.random.seed(2)
    mx.random.seed(2, ctx=ctx)

    train_transfroms = transforms.Compose([
        transforms.RandomBrightness(0.5),
        transforms.ToTensor()
    ])
    train_dataset = ImageDataset(config.trainfile, (config.img_h, config.img_w), 3, 80, config.alphabet, phase='train')
    train_data_loader = DataLoader(train_dataset.transform_first(train_transfroms), config.train_batch_size,
                                   shuffle=True,
                                   last_batch='keep', num_workers=config.workers)
    test_dataset = ImageDataset(config.testfile, (config.img_h, config.img_w), 3, 80, config.alphabet, phase='test')
    test_data_loader = DataLoader(test_dataset.transform_first(transforms.ToTensor()), config.eval_batch_size,
                                  shuffle=True,
                                  last_batch='keep', num_workers=config.workers)
    net = CRNN(len(config.alphabet), hidden_size=config.nh)
    net.hybridize()
    if not config.restart_training and config.checkpoint != '':
        logger.info('load pretrained net from {}'.format(config.checkpoint))
        net.load_parameters(config.checkpoint, ctx=ctx)
    else:
        net.initialize(ctx=ctx)

    criterion = gluon.loss.CTCLoss()

    all_step = len(train_data_loader)
    logger.info('each epoch contains {} steps'.format(all_step))
    schedule = mx.lr_scheduler.FactorScheduler(step=config.lr_decay_step * all_step, factor=config.lr_decay,
                                               stop_factor_lr=config.end_lr)
    # schedule = mx.lr_scheduler.MultiFactorScheduler(step=[15 * all_step, 30 * all_step, 60 * all_step,80 * all_step],
    #                                                 factor=0.1)
    adam_optimizer = mx.optimizer.Adam(learning_rate=config.lr, lr_scheduler=schedule)
    trainer = gluon.Trainer(net.collect_params(), optimizer=adam_optimizer)

    sw = SummaryWriter(logdir=config.output_dir)
    for epoch in range(config.start_epoch, config.end_epoch):
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

            loss_c = loss_ctc.mean().asscalar()
            cur_step = epoch * all_step + i
            sw.add_scalar(tag='ctc_loss', value=loss_c, global_step=cur_step // 2)
            sw.add_scalar(tag='lr', value=trainer.learning_rate, global_step=cur_step // 2)
            acc = accuracy(output, label, config.alphabet)
            train_acc += acc
            if (i + 1) % config.display_interval == 0:
                acc /= len(label)
                sw.add_scalar(tag='train_acc', value=acc, global_step=cur_step)
                batch_time = time.time() - tick
                logger.info(
                    '[{}/{}], [{}/{}],step: {}, Speed: {:.3f} samples/sec, ctc loss: {:.4f},acc: {:.4f}, lr:{},'
                    ' time:{:.4f} s'.format(epoch, config.end_epoch, i, all_step, cur_step,
                                            config.display_interval * config.train_batch_size / batch_time,
                                            loss_c, acc, trainer.learning_rate,
                                            batch_time))
                loss = .0
                tick = time.time()
                nd.waitall()
        if epoch == 0:
            sw.add_graph(net)
        train_acc /= train_dataset.__len__()
        validation_accuracy = evaluate_accuracy(net, test_data_loader, ctx, config.alphabet) / test_dataset.__len__()
        sw.add_scalar(tag='val_acc', value=validation_accuracy, global_step=cur_step)
        logger.info("Epoch {},train_acc {:.4f}, val_acc {:.4f}".format(epoch, train_acc, validation_accuracy))
        net.save_parameters(
            "{}/{}_{:.4f}_{:.4f}.params".format(config.output_dir, epoch, train_acc, validation_accuracy))
    sw.close()


if __name__ == '__main__':
    train()
