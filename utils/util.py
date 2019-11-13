# -*- coding: utf-8 -*-
# @Time    : 18-5-20 下午8:07
# @Author  : zhoujun
import time
import json
from collections import OrderedDict
from pathlib import Path


def setup_logger(log_file_path: str = None):
    import logging
    logger = logging.getLogger('crnn.gluon')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.info('logger init finished')
    return logger


# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back

    return newFunc


def save_json(data, json_path):
    with open(json_path, mode='w', encoding='utf8') as f:
        json.dump(data, f, indent=4)


def load_json(json_path):
    with open(json_path, mode='r', encoding='utf8') as f:
        data = json.load(f)
    return data


def read_json(fname):
    if isinstance(fname, str):
        fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)


def write_json(content, fname):
    if isinstance(fname, str):
        fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def try_gpu(gpu):
    import mxnet as mx
    from mxnet import nd
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def punctuation_mend(string):
    # 输入字符串或者txt文件路径
    import unicodedata
    import pathlib

    table = {ord(f): ord(t) for f, t in zip(
        u'，。！？【】（）％＃＠＆１２３４５６７８９０“”‘’',
        u',.!?[]()%#@&1234567890""\'\'')}  # 其他自定义需要修改的符号可以加到这里
    if pathlib.Path(string).is_file():
        with open(string, 'r', encoding='utf-8') as f:
            res = unicodedata.normalize('NFKC', f.read())
            res = res.translate(table)
        with open(string, 'w', encoding='utf-8') as f:
            f.write(res)
    else:
        res = unicodedata.normalize('NFKC', string)
        res = res.translate(table)
        return res


if __name__ == '__main__':
    print(punctuation_mend('1'))