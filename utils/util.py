# -*- coding: utf-8 -*-
# @Time    : 18-5-20 下午8:07
# @Author  : zhoujun
import time
import json
import pathlib
from tqdm import tqdm
from pathlib import Path


def setup_logger(log_file_path: str = None):
    import logging
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', )
    logger = logging.getLogger('crnn.gluon')
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s'))
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
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


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def _load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _save_txt, '.json': _save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def _save_txt(data, file_path):
    """
    将一个list的数组写入txt文件里
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def _save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def get_ctx(gpus):
    import mxnet as mx
    from mxnet import nd
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = []
        for gpu in gpus:
            ctx_i = mx.gpu(gpu)
            _ = nd.array([0], ctx=ctx_i)
            ctx.append(ctx_i)
    except:
        ctx = [mx.cpu()]
    return ctx


def punctuation_mend(string):
    # 输入字符串或者txt文件路径
    import unicodedata
    import pathlib

    table = {ord(f): ord(t) for f, t in zip(
        u'，。！？【】（）％＃＠＆１２３４５６７８９０“”‘’',
        u',.!?[]()%#@&1234567890""\'\'')}  # 其他自定义需要修改的符号可以加到这里
    res = unicodedata.normalize('NFKC', string)
    res = res.translate(table)
    return res


def get_datalist(data_path):
    """
    获取训练和验证的数据list
    :param data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    if isinstance(data_path, list):
        for p in data_path:
            train_data.extend(get_datalist(p))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc='load data from {}'.format(data_path)):
                line = line.strip('\n').replace('.jpg ', '.jpg\t').replace('.png ', '.png\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label = line[1]
                    if img_path.exists() and img_path.stat().st_size > 0:
                        train_data.append((str(img_path), label))
    return train_data


def parse_config(config: dict) -> dict:
    import anyconfig
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config


if __name__ == '__main__':
    print(punctuation_mend('1'))
