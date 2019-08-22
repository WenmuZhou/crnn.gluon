# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun
import torch
import time
import Levenshtein
from tqdm import tqdm
from mxnet import autograd

from base import BaseTrainer
from predict import decode


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, ctx, val_loader=None):
        super(Trainer, self).__init__(config, model, criterion, ctx)
        self.train_loader = train_loader
        self.train_loader_len = len(train_loader)
        self.val_loader_len = len(val_loader)
        self.val_loader = val_loader

        self.alphabet = self.config['data_loader']['args']['dataset']['alphabet']

        self.logger.info(
            'train dataset has {} samples,{} in dataloader, val dataset has {} samples,{} in dataloader'.format(
                self.train_loader.dataset_len,
                self.train_loader_len,
                self.val_loader.dataset_len,
                self.val_loader_len))

    def _train_epoch(self, epoch):
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        for i, (images, labels) in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            images = images.as_in_context(self.ctx)
            labels = labels.as_in_context(self.ctx)
            # 数据进行转换和丢到gpu
            cur_batch_size = images.shape[0]

            # forward
            with autograd.record():
                preds = self.model(images)
                loss = self.criterion(preds, labels)
            # backward
            loss.backward()
            self.trainer.step(cur_batch_size)

            # loss 和 acc 记录到日志
            loss = loss.mean().asscalar()
            train_loss += loss

            batch_dict = self.accuracy_batch(preds, labels, phase='TRAIN')
            acc = batch_dict['n_correct'] / cur_batch_size
            edit_dis = batch_dict['edit_dis'] / cur_batch_size

            if self.tensorboard_enable:
                # write tensorboard
                self.writer.add_scalar('TRAIN/ctc_loss', loss, self.global_step)
                self.writer.add_scalar('TRAIN/acc', acc, self.global_step)
                self.writer.add_scalar('TRAIN/edit_distance', edit_dis, self.global_step)
                self.writer.add_scalar('TRAIN/lr', self.trainer.learning_rate, self.global_step)

            if (i + 1) % self.display_interval == 0:
                batch_time = time.time() - batch_start
                self.logger.info(
                    '[{}/{}], [{}/{}], global_step: {}, Speed: {:.1f} samples/sec, ctc loss:{:.4f}, acc:{:.4f}, edit_dis:{:.4f} lr:{}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                                            self.display_interval * cur_batch_size / batch_time,
                        loss, acc, edit_dis, self.trainer.learning_rate, batch_time))
                batch_start = time.time()
        return {'train_loss': train_loss / self.train_loader_len, 'time': time.time() - epoch_start, 'epoch': epoch}

    def _eval(self):
        n_correct = 0
        edit_dis = 0
        for images, labels in tqdm(self.val_loader, desc='test model'):
            images = images.as_in_context(self.ctx)
            labels = labels.as_in_context(self.ctx)
            preds = self.model(images)
            batch_dict = self.accuracy_batch(preds, labels, phase='VAL')
            n_correct += batch_dict['n_correct']
            edit_dis += batch_dict['edit_dis']
        return {'n_correct': n_correct, 'edit_dis': edit_dis}

    def _on_epoch_finish(self):
        self.logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.trainer.learning_rate))

        save_best = False
        if self.val_loader is not None:
            epoch_eval_dict = self._eval()

            val_acc = epoch_eval_dict['n_correct'] / self.val_loader.dataset_len
            edit_dis = epoch_eval_dict['edit_dis'] / self.val_loader.dataset_len

            if self.tensorboard_enable:
                self.writer.add_scalar('EVAL/acc', val_acc, self.global_step)
                self.writer.add_scalar('EVAL/edit_distance', edit_dis, self.global_step)

            self.logger.info('[{}/{}], val_acc: {:.6f}'.format(self.epoch_result['epoch'], self.epochs, val_acc))

            net_save_path = '{}/CRNN_{}_loss{:.6f}_val_acc{:.6f}.params'.format(self.checkpoint_dir,
                                                                             self.epoch_result['epoch'],
                                                                             self.epoch_result['train_loss'],
                                                                             val_acc)
            if val_acc > self.metrics['val_acc']:
                save_best = True
                self.metrics['val_acc'] = val_acc
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model'] = net_save_path
        else:
            net_save_path = '{}/CRNN_{}_loss{:.6f}.params'.format(self.checkpoint_dir,
                                                               self.epoch_result['epoch'],
                                                               self.epoch_result['train_loss'])
            if self.epoch_result['train_loss'] < self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model'] = net_save_path
        self._save_checkpoint(self.epoch_result['epoch'], net_save_path, save_best)

    def accuracy_batch(self, predictions, labels, phase):
        predictions = predictions.softmax().asnumpy()
        zipped = zip(decode(predictions, self.alphabet), decode(labels.asnumpy(), self.alphabet))

        n_correct = 0
        edit_dis = 0.0
        logged = False
        for (pred, pred_conf), (target, _) in zipped:
            if self.tensorboard_enable and not logged:
                self.writer.add_text(tag='{}/pred'.format(phase),
                                     text='pred: {} -- gt:{}'.format(pred, target),
                                     global_step=self.global_step)
                logged = True
            edit_dis += Levenshtein.distance(pred, target)
            if pred == target:
                n_correct += 1
        return {'n_correct': n_correct, 'edit_dis': edit_dis}

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger.info('{}:{}'.format(k, v))
        self.logger.info('finish train')
