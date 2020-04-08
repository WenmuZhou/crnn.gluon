# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun
import time
import Levenshtein
from tqdm import tqdm
from mxnet import autograd
from mxnet.gluon import utils as gutils

from base import BaseTrainer
from predict import decode


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, ctx, sample_input, validate_loader=None):
        super().__init__(config, model, criterion, ctx, sample_input)
        self.train_loader = train_loader
        self.train_loader_len = len(train_loader)
        self.validate_loader = validate_loader

        if self.validate_loader is not None:
            self.logger.info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    self.train_loader.dataset_len, len(train_loader), self.validate_loader.dataset_len, len(self.validate_loader)))
        else:
            self.logger.info('train dataset has {} samples,{} in dataloader'.format(len(self.train_loader.dataset), len(self.train_loader)))

    def _train_epoch(self, epoch):
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        for i, (images, labels) in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            cur_batch_size = images.shape[0]
            # 将图片和gt划分到每个gpu
            gpu_images = gutils.split_and_load(images, self.ctx)
            gpu_labels = gutils.split_and_load(labels, self.ctx)
            # 数据进行转换和丢到gpu

            # forward
            with autograd.record():
                preds = [self.model(x)[0] for x in gpu_images]
                ls = [self.criterion(pred, gpu_y) for pred, gpu_y in zip(preds, gpu_labels)]
            # backward
            for l in ls:
                l.backward()
            self.trainer.step(cur_batch_size)

            # loss 和 acc 记录到日志
            loss = sum([x.sum().asscalar() for x in ls]) / sum([x.shape[0] for x in ls])
            train_loss += loss

            batch_dict = self.accuracy_batch(preds, gpu_labels, phase='TRAIN')
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
                    '[{}/{}], [{}/{}], global_step: {}, Speed: {:.1f} samples/sec, acc:{:.4f}, loss:{:.4f}, edit_dis:{:.4f} lr:{}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step, self.display_interval * cur_batch_size / batch_time,
                        acc, loss, edit_dis, self.trainer.learning_rate, batch_time))
                batch_start = time.time()
        return {'train_loss': train_loss / self.train_loader_len, 'time': time.time() - epoch_start, 'epoch': epoch}

    def _eval(self):
        n_correct = 0
        edit_dis = 0
        for images, labels in tqdm(self.validate_loader, desc='test model'):
            gpu_images = gutils.split_and_load(images, self.ctx)
            gpu_labels = gutils.split_and_load(labels, self.ctx)
            preds = [self.model(x)[0] for x in gpu_images]
            batch_dict = self.accuracy_batch(preds, gpu_labels, phase='VAL')
            n_correct += batch_dict['n_correct']
            edit_dis += batch_dict['edit_dis']
        return {'n_correct': n_correct, 'edit_dis': edit_dis}

    def _on_epoch_finish(self):
        self.logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.trainer.learning_rate))
        net_save_path = '{}/model_latest.params'.format(self.checkpoint_dir)

        save_best = False
        if self.validate_loader is not None:
            epoch_eval_dict = self._eval()

            val_acc = epoch_eval_dict['n_correct'] / self.validate_loader.dataset_len
            edit_dis = epoch_eval_dict['edit_dis'] / self.validate_loader.dataset_len

            if self.tensorboard_enable:
                self.writer.add_scalar('EVAL/acc', val_acc, self.global_step)
                self.writer.add_scalar('EVAL/edit_distance', edit_dis, self.global_step)

            self.logger.info('[{}/{}], val_acc: {:.6f}'.format(self.epoch_result['epoch'], self.epochs, val_acc))

            if val_acc >= self.metrics['val_acc']:
                save_best = True
                self.metrics['val_acc'] = val_acc
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model'] = net_save_path
        else:
            if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model'] = net_save_path
        self._save_checkpoint(self.epoch_result['epoch'], net_save_path, save_best)

    def accuracy_batch(self, predictions, labels, phase):
        n_correct = 0
        edit_dis = 0.0
        for gpu_prediction, gpu_label in zip(predictions, labels):
            gpu_prediction = gpu_prediction.softmax().asnumpy()
            zipped = zip(decode(gpu_prediction, self.alphabet), decode(gpu_label.asnumpy(), self.alphabet))
            logged = False
            for (pred, pred_conf), (target, _) in zipped:
                if self.tensorboard_enable and not logged:
                    self.writer.add_text(tag='{}/pred'.format(phase), text='pred: {} -- gt:{}'.format(pred, target), global_step=self.global_step)
                    logged = True
                edit_dis += Levenshtein.distance(pred, target)
                if pred == target:
                    n_correct += 1
        return {'n_correct': n_correct, 'edit_dis': edit_dis}

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger.info('{}:{}'.format(k, v))
        self.logger.info('finish train')
